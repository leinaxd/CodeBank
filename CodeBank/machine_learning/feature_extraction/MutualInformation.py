# https://www.youtube.com/watch?v=YKDZHPJ-pQ0
# Se puede utilizar la informacion mutua, como medida de 
#   - Relevancia
#   - Redundancia
#
#   Features Extraction
#   -> Encontrar un conjunto compacto de features que sean lo más informativas posibles.
#   -> Medir la información mutua, entre la clase y diferentes conjuntos de features
#   -> max I(set_1, set_2) 
#   -> max I(C, {fi, fj})
#   -> Estima relevancia, y redundancia

#   Algorithm
#   -> mRMR
#       Max Relevance - Min Redundancy
#   
#       max D(S,c)
#           D = 1/|S| sum_{xi \in S}  I(xi;c)
#       min R(S)
#           R = 1/|S|^2 sum_{xi , xj \in S} I(xi ; xj)
#       paper:
#           Feature Selection Based on Mutual Information: 
#           Criteria of Max Dependency, Max relevance and Min Redundancy
#           ~Hanchuang Peng, Fuhui Long, Chris Ding
#           
#           max \PHI
#               \Phi = D-R
#           max_{xj \in X-S_{m-1}} [I(xj ; c) - 1/(m-1) \sum_{xi \in S_{m-1}} I(xj ; xi)]
#       Similar: 
#           MIFS
#           MIEF
#           Correlación (es para normales / lineales)
#       Mejoras:
#           KNN / parsen window / Kernell density estimation
