from heapq import heappush, heappop

VERBOSE = False


def heapsched_rt(lrts, k):
    h = []
    for rt in lrts[:k]:
        heappush(h, rt)

    curr_rt = 0
    for rt in lrts[k:]:
        curr_rt = heappop(h)
        heappush(h, rt + curr_rt)

    while len(h) > 0:
        curr_rt = heappop(h)

    return curr_rt


def parallelized_rt(lrts, k):
    if len(lrts) == 0:
        return 0.0
    inorder_rt = heapsched_rt(lrts, k)
    cp_bound = max(lrts)
    area_bound = sum(lrts) / k
    lrts.sort(reverse=True)
    two_approx = heapsched_rt(lrts, k)

    if VERBOSE:
        print("-- in incoming order, schedule= ", inorder_rt)
        print("-- bounds cp= ", cp_bound, "; area= ", area_bound)
        print("-- sorted rts: ", lrts)
        print("-- in sorted order, schedule ", two_approx)

    return two_approx
