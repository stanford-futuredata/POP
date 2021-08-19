import numpy as np
import heapq
import copy


class MaxHeapObj(object):
    def __init__(self, entity):
        self.entity = entity

    # reverse comparison of demands, since heapq implemented as minheap
    def __lt__(self, other):
        return self.entity[-1] > other.entity[-1]

    def __eq__(self, other):
        return self.entity == other

    def __str__(self):
        return str(self.entity)

    def get_entity(self):
        return self.entity

    def split(self, factor):
        self.entity[-1] = self.entity[-1] * factor


def halve(entity_mho):
    halved_entities_mho = [
        MaxHeapObj(copy.deepcopy(entity_mho.get_entity())) for _ in range(2)
    ]
    for entity_mho in halved_entities_mho:
        entity_mho.split(1 / 2.0)

    return halved_entities_mho


# Split the max entities in half (on last dimension)
# until add_fraction new entities are formed.
# Return a list of lists containing all splits of each entity
def split_entities(entity_list, add_fraction):

    print("splitting for additional " + str(add_fraction) + " entities")

    num_entities = len(entity_list)
    num_new_entities = np.round(num_entities * add_fraction)

    # use dict to keep track of all splits of each original entity
    entity_splits_dict = {}
    for entity in entity_list:
        entity_splits_dict[tuple(entity[:-1])] = [entity]

    # creat MaxHeapObject list of entities
    entity_mho_list = [MaxHeapObj(entity) for entity in entity_list]

    # create heap of maxHeapObjects (a maxheap)
    heapq.heapify(entity_mho_list)

    while len(entity_mho_list) < num_entities + num_new_entities:
        largest_entity_mho = heapq.heappop(entity_mho_list)

        # split largest entity
        new_entities = halve(largest_entity_mho)

        # add it to heap
        for new_entity in new_entities:
            heapq.heappush(entity_mho_list, new_entity)

        # update splits dict
        entity = largest_entity_mho.get_entity()
        entity_splits_dict[tuple(entity[:-1])].remove(entity)
        for new_entity in new_entities:
            entity_splits_dict[tuple(entity[:-1])].append(new_entity.get_entity())

    # new_entity_list = [entity_mho.get_entity() for entity_mho in entity_mho_list]

    # group together all entities that share id
    grouped_entity_list = [
        entity_splits_dict[tuple(entity[:-1])] for entity in entity_list
    ]

    return grouped_entity_list
