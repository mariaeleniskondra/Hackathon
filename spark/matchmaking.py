"""Business matchmaking με Spark DataFrames (rule-based scoring, όχι εκπαιδευμένο μοντέλο).

Βήματα:
 1. Offers = skills ∪ services, Needs = needs (κανονικοποιημένα).
 2. Υποψήφια ζεύγη μέσω explode(needs) ⨝ explode(offers) στον κοινό όρο, ώστε να ΜΗΝ
    δημιουργούνται όλα τα ζεύγη. Κάθε forward ζεύγος (A,B) δίνει υποψήφιο και προς τις δύο
    κατευθύνσεις (A,B) και (B,A), ώστε να πιάνεται και η μονόπλευρη κάλυψη από την άλλη μεριά.
 3. Scoring (scoring.py), εξαίρεση self-match, Top-K ανά προφίλ.
 4. matched_skills = οι όροι που ο partner προσφέρει και το προφίλ χρειάζεται (forward)·
    match_reason = σύντομη εξήγηση και για τις δύο κατευθύνσεις.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from scoring import TOP_K, category_score, coverage, location_score, total_score


def _norm_list(col):
    """lower + trim, αφαίρεση κενών και διπλών. Ο loader της Μιλένας δεν κανονικοποιεί τους
    όρους (κρατά ό,τι έχει το JSON), οπότε το κάνει εδώ το matchmaking. Η κανονική
    κανονικοποίηση συνωνύμων γίνεται upstream (normalizer Μαριφύλλης)· εδώ είναι δίχτυ ασφαλείας."""
    cleaned = F.transform(F.coalesce(col, F.array().cast("array<string>")), lambda x: F.lower(F.trim(x)))
    return F.array_distinct(F.filter(cleaned, lambda x: x != ""))


def prepare_profiles(profiles: DataFrame) -> DataFrame:
    """Κανονικοποιεί category/location (πεζά, trim) και skills/services/needs.
    Δουλεύει πάνω στο schema του κοινού data_loader.py (id, name, category, location, skills, ...).
    Είναι idempotent: μπορεί να κληθεί και σε ήδη κανονικοποιημένα δεδομένα."""
    return (
        profiles.filter(F.col("id").isNotNull())
        .dropDuplicates(["id"])
        .select(
            "id",
            "name",
            F.lower(F.trim("category")).alias("category"),
            F.lower(F.trim("location")).alias("location"),
            _norm_list(F.col("skills")).alias("skills"),
            _norm_list(F.col("services")).alias("services"),
            _norm_list(F.col("needs")).alias("needs"),
        )
    )


def _with_offers(profiles: DataFrame) -> DataFrame:
    return profiles.select(
        "id",
        "name",
        "category",
        "location",
        "needs",
        F.array_distinct(F.array_union("skills", "services")).alias("offers"),
    )


def candidate_pairs(p: DataFrame) -> DataFrame:
    """Διακριτά (a_id, b_id) όπου ο b προσφέρει κάτι που ο a χρειάζεται, ή το αντίστροφο."""
    needs = p.select(F.col("id").alias("a_id"), F.explode("needs").alias("term"))
    offers = p.select(F.col("id").alias("b_id"), F.explode("offers").alias("term"))
    forward = needs.join(offers, "term").where(F.col("a_id") != F.col("b_id")).select("a_id", "b_id")
    both = forward.unionByName(forward.select(F.col("b_id").alias("a_id"), F.col("a_id").alias("b_id")))
    return both.distinct()


def compute_partners(profiles: DataFrame, top_k: int = TOP_K) -> DataFrame:
    p = _with_offers(prepare_profiles(profiles))
    pairs = candidate_pairs(p)

    a = p.select(
        F.col("id").alias("a_id"),
        F.col("name").alias("a_name"),
        F.col("category").alias("a_cat"),
        F.col("location").alias("a_loc"),
        F.col("needs").alias("a_needs"),
        F.col("offers").alias("a_offers"),
    )
    b = p.select(
        F.col("id").alias("b_id"),
        F.col("name").alias("b_name"),
        F.col("category").alias("b_cat"),
        F.col("location").alias("b_loc"),
        F.col("needs").alias("b_needs"),
        F.col("offers").alias("b_offers"),
    )

    j = pairs.join(a, "a_id").join(b, "b_id")

    fwd_terms = F.array_sort(F.array_intersect("a_needs", "b_offers"))  # b προσφέρει όσα χρειάζεται ο a
    rev_terms = F.array_sort(F.array_intersect("b_needs", "a_offers"))  # a προσφέρει όσα χρειάζεται ο b

    scored = (
        j.withColumn("forward", coverage(F.col("a_needs"), F.col("b_offers")))
        .withColumn("reverse", coverage(F.col("b_needs"), F.col("a_offers")))
        .withColumn("matched_skills", fwd_terms)
        .withColumn("reverse_terms", rev_terms)
        .withColumn(
            "match_score",
            total_score(
                F.col("forward"),
                F.col("reverse"),
                category_score(F.col("a_cat"), F.col("b_cat")),
                location_score(F.col("a_loc"), F.col("b_loc")),
            ),
        )
        # τουλάχιστον μία πραγματική ανάγκη καλύπτεται προς μία κατεύθυνση
        .where((F.size("matched_skills") > 0) | (F.size("reverse_terms") > 0))
    )

    reason = F.concat_ws(
        "; ",
        F.when(F.size("matched_skills") > 0, F.concat(F.lit("Offers "), F.array_join("matched_skills", ", "))),
        F.when(
            F.size("reverse_terms") > 0,
            F.concat(F.lit("Needs "), F.array_join("reverse_terms", ", "), F.lit(" which you offer")),
        ),
    )

    w = Window.partitionBy("a_id").orderBy(F.col("match_score").desc(), F.col("b_id").asc())
    return (
        scored.withColumn("rank_position", F.row_number().over(w))
        .where(F.col("rank_position") <= top_k)
        .select(
            F.col("a_id").alias("profile_id"),
            F.col("b_id").alias("partner_id"),
            F.col("b_name").alias("partner_name"),
            "match_score",
            "matched_skills",
            reason.alias("match_reason"),
            "rank_position",
            F.current_timestamp().alias("computed_at"),
        )
        .orderBy("profile_id", "rank_position")
    )
