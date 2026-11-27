"""Συμβόλαιο scoring για το matchmaking (V3 §10).

Score = 0.50*forward + 0.30*reverse + 0.10*category + 0.10*location

forward(A,B) = |Needs(A) ∩ Offers(B)| / |Needs(A)|   (0 αν Needs(A) κενό)
reverse(A,B) = |Needs(B) ∩ Offers(A)| / |Needs(B)|   (0 αν Needs(B) κενό)
Offers(X)    = skills(X) ∪ services(X)   (products ΟΧΙ, εκτός αν το taxonomy ορίσει αντιστοίχιση)

Κανόνες:
 * self-match αποκλείεται
 * κρατιέται υποψήφιος μόνο αν καλύπτει >= 1 πραγματική ανάγκη προς μία από τις δύο κατευθύνσεις
   (κλάδος/περιοχή μόνα τους δεν αρκούν)
 * Top-K ανά προφίλ, ταξινόμηση score DESC, partner_id ASC
"""
from pyspark.sql import Column
from pyspark.sql import functions as F

W_FORWARD = 0.50
W_REVERSE = 0.30
W_CATEGORY = 0.10
W_LOCATION = 0.10

TOP_K = 5
SCORE_DECIMALS = 6  # στρογγυλοποίηση ώστε οι ισοβαθμίες να είναι ντετερμινιστικές

# Συμφωνημένα συμβατά ζεύγη κλάδων (πεζά). Ίδιος κλάδος = πάντα συμβατός.
# ΠΡΟΣΩΡΙΝΟ: να οριστεί με την ομάδα βάσει του taxonomy.json της Μαριφύλλης.
COMPATIBLE_CATEGORIES = {
    frozenset({"technology", "marketing"}),
    frozenset({"technology", "design"}),
    frozenset({"marketing", "design"}),
}


def coverage(needs: Column, offers: Column) -> Column:
    """Ποσοστό των needs που καλύπτονται από offers. 0 για κενά needs."""
    n = F.size(needs)
    return F.when(n == 0, F.lit(0.0)).otherwise(F.size(F.array_intersect(needs, offers)) / n)


def category_score(cat_a: Column, cat_b: Column) -> Column:
    cond = cat_a.isNotNull() & cat_b.isNotNull() & (cat_a == cat_b)
    for pair in COMPATIBLE_CATEGORIES:
        x, y = sorted(pair)
        cond = cond | ((cat_a == F.lit(x)) & (cat_b == F.lit(y))) | ((cat_a == F.lit(y)) & (cat_b == F.lit(x)))
    return F.when(cond, F.lit(1.0)).otherwise(F.lit(0.0))


def location_score(loc_a: Column, loc_b: Column) -> Column:
    return F.when(loc_a.isNotNull() & loc_b.isNotNull() & (loc_a == loc_b), F.lit(1.0)).otherwise(F.lit(0.0))


def total_score(forward: Column, reverse: Column, category: Column, location: Column) -> Column:
    s = W_FORWARD * forward + W_REVERSE * reverse + W_CATEGORY * category + W_LOCATION * location
    return F.round(s, SCORE_DECIMALS)
