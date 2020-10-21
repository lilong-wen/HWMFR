# HWMFR
Handwritten math formula recognition

###TODO
. remove chinese character in pre_process using "line = re.sub("([^\x00-\x7F])+", "", line)"
. if chinese character removed, add this line in the dataIter function "while ("" in tmp): tmp.remove("")"