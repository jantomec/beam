import numpy as np
import elements as ele


def identify_entities(elements):
    beams = []
    beam = set()
    for i in range(len(elements)-1):
        if type(elements[i]) == ele.SimoBeam and type(elements[i+1]) == ele.SimoBeam:
            if elements[i].nodes[-1] == elements[i+1].nodes[0]:
                beam.add(i)
            else:
                beam.add(i)
                beams.append(beam)
                beam = set()
    beam.add(i+1)
    beams.append(beam)
    return beams

def collect_nodes(elements):
    nodes = set()
    for e in elements:
        nodes = nodes | set(e.nodes)
    return np.array(list(nodes), dtype=np.int)