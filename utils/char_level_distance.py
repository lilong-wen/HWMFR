import numpy

def edit_distance(label,rec):
    """compute char level distance from label to recognized result
    args:
        label: truth label
        rec: recognized result
    returns:
        dist: character level distance from label to rec
        len(label): length of label
    """

    dist_mat = numpy.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    #print(dist_mat)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)
    #print(dist_mat)
    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)


# for test
if __name__  == "__main__":
    label = "abc def"
    rec = "abc dnf"

    dist, l = edit_distance(label, rec)
    print(dist)
    print(lz)

