"""
Module containing the core concept transformation algebra. Usage:

    >>> from quangis.transformation.cct import cct
    >>> expr = cct.parse("pi1 (objects data)")
    >>> print(expr.type)
    R(Obj)
"""

from quangis.transformation.type import TypeOperator, Variables, Typeclass, Constraint
from quangis.transformation.algebra import TransformationAlgebra

cct = TransformationAlgebra()
var = Variables()

##############################################################################
# Types and type synonyms

Val = TypeOperator("Val")
Obj = TypeOperator("Obj", supertype=Val)  # O
Reg = TypeOperator("Reg", supertype=Val)  # S
Loc = TypeOperator("Loc", supertype=Val)  # L
Qlt = TypeOperator("Qlt", supertype=Val)  # Q
Nom = TypeOperator("Nom", supertype=Qlt)
Bool = TypeOperator("Bool", supertype=Nom)
Ord = TypeOperator("Ord", supertype=Nom)
Itv = TypeOperator("Itv", supertype=Ord)
Ratio = TypeOperator("Ratio", supertype=Itv)
Count = TypeOperator("Count", supertype=Ratio)
R1 = TypeOperator.parameterized("R1", 1)  # Collections
R2 = TypeOperator.parameterized("R2", 2)  # Unary core concepts, 1 key (left)
R3 = TypeOperator.parameterized("R3", 3)  # Quantified relation, 2 keys (l & r)
R3a = TypeOperator.parameterized("R3a", 3)  # Ternary relation, 1 key (left)

SpatialField = R2(Loc, Qlt)
InvertedField = R2(Qlt, Reg)
FieldSample = R2(Reg, Qlt)
ObjectExtent = R2(Obj, Reg)
ObjectQuality = R2(Obj, Qlt)
NominalField = R2(Loc, Nom)
BooleanField = R2(Loc, Bool)
NominalInvertedField = R2(Nom, Reg)
BooleanInvertedField = R2(Bool, Reg)

##############################################################################
# Typeclasses

subtype = Typeclass("Subtype", var.subtype, var.type)
subtype.instance(Obj, Val)
subtype.instance(Reg, Val)
# etc

param1 = Typeclass("Param1", var.relationship, var.param, dependent=[var.param])
param1.instance(R1(var.x), var.x)
param1.instance(R2(var.x, var._), var.x)
param1.instance(R3(var.x, var._, var._), var.x)

param2 = Typeclass("Param2", var.relationship, var.param, dependent=[var.param])
param2.instance(R2(var._, var.x), var.x)
param2.instance(R3(var._, var.x, var._), var.x)

param3 = Typeclass("Param3", var.relationship, var.param, dependent=[var.param])
param3.instance(R3(var._, var._, var.x), var.x)

param = Typeclass("Param", var.relationship, var.param)
param.instance(var.rel, var.x, constraints=[param1(var.rel, var.x)])
param.instance(var.rel, var.x, constraints=[param2(var.rel, var.x)])
param.instance(var.rel, var.x, constraints=[param3(var.rel, var.x)])
# etc


# Ad-hoc extensional typeclass constructor
def member(v, options):
    typeclass = Typeclass("membership")
    for o in options:
        typeclass.instance(o)
    return Constraint(typeclass, v)


##############################################################################
# Data inputs

cct.pointmeasures = R2(Reg, Itv), 1
cct.amountpatches = R2(Reg, Nom), 1
cct.countamounts = R2(Reg, Count), 1
cct.boolcoverages = R2(Bool, Reg), 1
cct.boolratio = R2(Bool, Ratio), 1
cct.nomcoverages = R2(Nom, Reg), 1
cct.nomsize = R2(Nom, Ratio), 1
cct.regions = R1(Reg), 1
cct.contour = R2(Ord, Reg), 1
cct.objectratios = R2(Obj, Ratio), 1
cct.objectnominals = R2(Obj, Nom), 1
cct.objectregions = R2(Obj, Reg), 1
cct.contourline = R2(Itv, Reg), 1
cct.objectcounts = R2(Obj, Count), 1
cct.field = R2(Loc, Ratio), 1
cct.nomfield = R2(Loc, Nom), 1
cct.boolfield = R2(Loc, Bool), 1
cct.ordfield = R2(Loc, Ord), 1
cct.itvfield = R2(Loc, Itv), 1
cct.object = Obj, 1
cct.objects = R1(Obj), 1
cct.region = Reg, 1
cct.in_ = Nom, 0
cct.out = Nom, 0
cct.noms = R1(Nom), 1
cct.ratios = R1(Ratio), 1
cct.countV = Count, 1
cct.ratioV = Ratio, 1
cct.interval = Itv, 1
cct.ordinal = Ord, 1
cct.nominal = Nom, 1
cct.true = Bool, 0

###########################################################################
# Math/stats transformations

# functions to handle multiple attributes of the same types with 1 key
cct.join_attr = R2(var.x, var.y) ** R2(var.x, var.z) ** R3a(var.x, var.y, var.z)
cct.get_attrL = R3a(var.x, var.y, var.z) ** R2(var.x, var.y)
cct.get_attrR = R3a(var.x, var.y, var.z) ** R2(var.x, var.z)

# functional
cct.compose = (var.y ** var.z) ** (var.x ** var.y) ** (var.x ** var.z)
cct.swap = (var.x ** var.y ** var.z) ** (var.y ** var.x ** var.z)
cct.cast = var.x ** var.y, subtype(var.x, var.y)

# derivations
cct.ratio = Ratio ** Ratio ** Ratio
cct.product = Ratio ** Ratio ** Ratio
cct.leq = var.x ** var.x ** Bool, subtype(var.x, Ord)
cct.eq = var.x ** var.x ** Bool, subtype(var.x, Val)
cct.conj = Bool ** Bool ** Bool
cct.disj = Bool ** Bool ** Bool  # define as not-conjunction
cct.notj = Bool ** Bool

# aggregations of collections
cct.count = R1(Obj) ** Ratio
cct.size = R1(Loc) ** Ratio
cct.merge = R1(Reg) ** Reg
cct.centroid = R1(Loc) ** Loc
cct.name = R1(Nom) ** Nom

# statistical operations
cct.avg = R2(var.v, var.x) ** var.x, subtype(var.v, Val), subtype(var.x, Itv)
cct.min = R2(var.v, var.x) ** var.x, subtype(var.v, Val), subtype(var.x, Ord)
cct.max = R2(var.v, var.x) ** var.x, subtype(var.v, Val), subtype(var.x, Ord)
cct.sum = R2(var.v, var.x) ** var.x, subtype(var.v, Val), subtype(var.x, Ratio)
# define in terms of: nest2 (merge pi1) (sum)
cct.contentsum = R2(Reg, var.x) ** R2(Reg, var.x), subtype(var.x, Ratio)
# define in terms of: nest2 (name pi1) (sum)
cct.coveragesum = R2(var.v, var.x) ** R2(Nom, var.x), subtype(var.x, Ratio), subtype(var.v, Nom)


##########################################################################
# Geometric transformations

cct.interpol = R2(Reg, Itv) ** R1(Loc) ** R2(Loc, Itv)
# should be defined with ldist somehow
cct.extrapol = R2(Obj, Reg) ** R2(Loc, Bool)  # Buffering
cct.arealinterpol = R2(Reg, Ratio) ** R1(Reg) ** R2(Reg, Ratio)

# deify/reify, nest/get, invert/revert might be defined in terms of inverse
cct.inverse = (var.x ** var.y) ** (var.y ** var.x)

# conversions
cct.reify = R1(Loc) ** Reg
cct.deify = Reg ** R1(Loc)
cct.nest = var.x ** R1(var.x)  # Puts values into some unary relation
cct.nest2 = var.x ** var.y ** R2(var.x, var.y)
cct.nest3 = var.x ** var.y ** var.z ** R3(var.x, var.y, var.z)
cct.get = R1(var.x) ** var.x, subtype(var.x, Val)
cct.invert = R2(Loc, var.x) ** R2(var.x, Reg), subtype(var.x, Qlt)
cct.revert = R2(var.x, Reg) ** R2(Loc, var.x), subtype(var.x, Qlt)
# could be definable with a projection operator that is applied to ternary
# relation (?)
cct.getamounts = R2(Obj, var.x) ** R2(Obj, Reg) ** R2(Reg, var.x), subtype(var.x, Ratio)

# quantified relations
# define odist in terms of the minimal ldist
cct.oDist = R2(Obj, Reg) ** R2(Obj, Reg) ** R3(Obj, Ratio, Obj)
cct.lDist = R1(Loc) ** R1(Loc) ** R3(Loc, Ratio, Loc)
# similar for lodist
cct.loDist = R1(Loc) ** R2(Obj, Reg) ** R3(Loc, Ratio, Obj)
cct.oTopo = R2(Obj, Reg) ** R2(Obj, Reg) ** R3(Obj, Nom, Obj)
cct.loTopo = R1(Loc) ** R2(Obj, Reg) ** R3(Loc, Nom, Obj)
# otopo can be defined in terms of rtopo? in rtopo, if points of a region are
# all inside, then the region is inside
cct.rTopo = R1(Reg) ** R1(Reg) ** R3(Reg, Nom, Reg)
cct.lTopo = R1(Loc) ** R1(Loc) ** R3(Loc, Nom, Loc)
cct.nDist = R1(Obj) ** R1(Obj) ** R3(Obj, Ratio, Obj) ** R3(Obj, Ratio, Obj)
cct.lVis = R1(Loc) ** R1(Loc) ** R2(Loc, Itv) ** R3(Loc, Bool, Loc)

# amount operations
cct.fcont = (R2(var.v, var.x) ** var.x) ** R2(Loc, var.x) ** Reg ** Ratio, subtype(var.x, Qlt), subtype(var.v, Val)
cct.ocont = R2(Obj, Reg) ** Reg ** Count
cct.fcover = R2(Loc, var.x) ** R1(var.x) ** Reg, subtype(var.x, Qlt)
cct.ocover = R2(Obj, Reg) ** R1(Obj) ** Reg


###########################################################################
# Relational transformations

cct.apply = R2(var.x, var.y) ** var.x ** var.y

# Projection (π). Projects a given relation to one of its attributes,
# resulting in a collection.
cct.pi1 = var.rel ** R1(var.x), param1(var.rel, var.x)
cct.pi2 = var.rel ** R1(var.x), param2(var.rel, var.x)
cct.pi3 = var.rel ** R1(var.x), param3(var.rel, var.x)

# Selection (σ). Selects a subset of the relation using a constraint on
# attribute values, like equality (eq) or order (leq). Used to be sigmae
# and sigmale.
cct.select = (
    (var.x ** var.y ** Bool) ** var.rel ** var.y ** var.rel,
    param(var.rel, var.x1), subtype(var.x1, var.x)
)

# Join of two unary concepts, like a table join.
# is join the same as join_with2 eq?
cct.join = R2(var.x, var.y) ** R2(var.y, var.z) ** R2(var.x, var.z)

# Join on subset (⨝). Subset a relation to those tuples having an attribute
# value contained in a collection. Used to be bowtie.
cct.join_subset = (
    var.rel ** R1(var.x) ** var.rel,
    param(var.rel, var.x)
)

# Join (⨝*). Substitute the quality of a quantified relation to some
# quality of one of its keys. Used to be bowtie*.
cct.join_key = (
    R3(var.x, var.q1, var.y) ** var.rel ** R3(var.x, var.q2, var.y),
    member(var.rel, [R2(var.x, var.q2), R2(var.y, var.q2)])
)

# Join with unary function. Generate a unary concept from one other unary
# concept of the same type. Used to be join_fa.
cct.join_with1 = (
    (var.x1 ** var.x2)
    ** R2(var.y, var.x1) ** R2(var.y, var.x2)
)

# Join with binary function (⨝_f). Generate a unary concept from two other
# unary concepts of the same type. Used to be bowtie_ratio and others.
cct.join_with2 = (
    (var.x1 ** var.x2 ** var.x3)
    ** R2(var.y, var.x1) ** R2(var.y, var.x2) ** R2(var.y, var.x3)
)

# Group by (β). Group quantified relations by the left (right) key,
# summarizing lists of quality values with the same key value into a new
# value per key, resulting in a unary core concept relation.
cct.groupbyL = (
    (var.rel ** var.q2) ** R3(var.l, var.q1, var.r) ** R2(var.l, var.q2),
    member(var.rel, [R1(var.r), R2(var.r, var.q1)])
)

cct.groupbyR = (
    (var.rel ** var.q2) ** R3(var.l, var.q1, var.r) ** R2(var.r, var.q2),
    member(var.rel, [R1(var.l), R2(var.l, var.q1)])
)
