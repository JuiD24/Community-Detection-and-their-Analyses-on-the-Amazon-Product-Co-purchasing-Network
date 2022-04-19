# Jui Dakhave, Saurabh Kulkarni, Rajath W
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import itertools
import random
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import random
from networkx.algorithms.community import modularity
from networkx.generators.community import LFR_benchmark_graph
import community.community_louvain
from scipy import stats

# Methods to draw communities start
def community_layout(g, partition):
    """ Compute the layout for a modular graph.
    """
    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """Positions nodes within communities.
    """
    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

# Methods to draw communities end

# Building graph of 30k nodes
def buildGraph():
    G1 = nx.Graph()
    G1 = nx.read_adjlist("sampled_graph_30k.adjlist")
    return G1

# Returns the original graph after building it from txt file.
def getOriginalGraph():
	file1 = open('com-amazon.ungraph.txt', 'r')
	Lines = file1.readlines()
	G = nx.Graph()
	for line in Lines:
		a, b = line.strip().split("\t")
		G.add_edge(a, b)
	return G

# Build mapping between nodes and product ids
def createGraphNodesMapping(): 
	G = getOriginalGraph()
	mapping = {v:k for k, v in enumerate(list(G.nodes))}
	return mapping


# Builds an adjcency list from original grpah
def buildAdjList():

	mapping = createGraphNodesMapping()
	
	Gnew = nx.Graph()
	file = open('com-amazon.ungraph.txt', 'r')
	Lines = file.readlines()
	
	for line in Lines:
		a, b = line.strip().split("\t")
		Gnew.add_edge(mapping.get(int(a)), mapping.get(int(b)))
	
	G1 = communitySampling(Gnew, 30000)
	nx.write_adjlist(G1, "sampled_graph_30k.adjlist")

# Degree distribution of the network
def plotDegDist():
    G = buildGraph()
    n = G.number_of_nodes()
    degreeFreq = {}
    degrees = [G.degree(node) for node in G.nodes()]
    for degree in degrees:
        degreeFreq[degree] = degreeFreq.get(degree, 0) + 1
    (X, Y) = zip(*[(degree, degreeFreq[degree] / n) for degree in degreeFreq])
    plt.scatter(X, Y, c='b', alpha=0.5)
    plt.title('Degree Distribution')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Density')
    plt.savefig('degreeDistribution.png')


# Selecting best communities from ground-truth communities
def buildCommunities():
    idToNodeMap = createGraphNodesMapping()

    file3 = open('com-amazon.all.dedup.cmty.txt', 'r')
    Lines = file3.readlines()

    communities=[]
    for i in range(75150):
        communities.append([])
    i=0
    for line in Lines:
        a = line.strip().split("\t")
        for n in a:
            communities[i].append(idToNodeMap[int(n)])
        i+=1

    for i, comm in enumerate(communities):
        communities[i] = set(comm)

    nodes = list(idToNodeMap.values())
    nodeCommMapping = {}
    for node in nodes:
        prevLen = -1
        for i, comm in enumerate(communities):
            if node in comm:
                if len(comm) > prevLen:
                    nodeCommMapping[node] = i
                    prevLen = len(comm)

    f = open("nodeCommunities.txt", "a")    
    for key in nodeCommMapping:
        f.write(str(key) + ' ' + str(nodeCommMapping[key]) + '\n')
    f.close()

# <Section - Detecting communities using Greedy modularity, Girvan newman and Louvain algorithm>
def greedyModularityCommunities(G):
    partition = greedy_modularity_communities(G)
    part = []
    i=0
    a={}
    for s in partition:
        for x in s:
            part.append(x)
            a[x]=i
        i += 1
  
    pos = community_layout(G, a)
    figure(figsize = (10, 8), dpi = 100)
    nx.draw(G, pos, node_color=list(a.values()))
    plt.show()

def girvanNewmanCommunityDetection(G):
	girvanNewmanCommunities = nx.algorithms.community.centrality.girvan_newman(G)
	max_modularity = float('-inf')
	bestPartition = tuple()

	for communities in itertools.islice(girvanNewmanCommunities, 2):
		a = tuple(c for c in communities)
		Q = modularity(G, a)
		if Q > max_modularity:
			max_modularity = Q
			bestPartition = a
	i=0
	a={}
	for s in bestPartition:
		for x in s:
			a[x]=i
		i += 1
	pos = community_layout(G, a)
	figure(figsize = (10, 8), dpi = 100)
	nx.draw(G, pos, node_color=list(a.values()))
	plt.show()

def louvainCommunities(G):
    partition = community.community_louvain.best_partition(G)

    pos = community_layout(G, partition)
    figure(figsize = (10, 8), dpi = 100)
    nx.draw(G, pos, node_color=list(partition.values()))
    plt.show()

# <Section - Accuracy for Louvain and Greedy Modularity>
def getGroundTruthCommList():
    G1 = buildGraph()
    nodes = set(G1.nodes())
    f = open('community_nodes_30k.txt', 'r')
    lines = f.readlines()
    communityList = []
    for line in lines:
        split = line.strip().split('->')
        nodesInComm = split[1].strip().split(' ')
        label = split[0]
        for node in nodesInComm:
            if node in nodes:
                communityList.append(label)
    return communityList

def getGroundTruthNodes():
    G1 = buildGraph()
    nodes = set(G1.nodes())
    f = open('node_comm_mapping_all.txt', 'r')
    lines = f.readlines()
    nodeList = []
    for line in lines:
        split = line.strip().split(' ')
        if split[0] in nodes:
            nodeList.append(split[0])
    return nodeList

def louvainCommDetNMI():
	G1 = buildGraph()
	partition = community.community_louvain.best_partition(G1)

	groundTruthCommList = getGroundTruthCommList()
	groundTruthNodes = set(getGroundTruthNodes())

	communityList=[]
	for nodes in G1.nodes():
		if nodes in groundTruthNodes:
			communityList.append(partition[nodes])
            
	NMI = normalized_mutual_info_score(groundTruthCommList, communityList)

def greedyCommDetNMI():
    G1 = buildGraph()
    greedyModularity = greedy_modularity_communities(G1)
    
    groundTruthCommList = getGroundTruthCommList()
    groundTruthNodes = set(getGroundTruthNodes())

    communityList=[]
    for node in G1.nodes():
        i=0
        for x in greedyModularity:
            if node in x and node in groundTruthNodes:                
                communityList.append(i)
                break
            i+=1
    NMI = normalized_mutual_info_score(groundTruthCommList, communityList)

# <Section - LFR ------------------------------------------------------------>
def getTau1(G):
  degrees = {}
  degreeList = [G.degree(v) for v in G.nodes()]
  for deg in degreeList:
    degrees[deg] = degrees.get(deg, 0) + 1

  (X, Y) = zip(*[(key,degrees[key]/len(G)) for key in degrees]) 
  (logX, logY) = ([np.log10(x) for x in X], [np.log10(y) for y in Y])

  resultLogLog = stats.linregress(logX, logY)
  return resultLogLog.slope*-1

def getTau2(label_to_nodeList):
  X=[]
  Y=[]
  communityLengths = []
  for key in label_to_nodeList:
    length = len(label_to_nodeList[key])
    communityLengths.append(length)
    
  communityDist = {}
  for n in communityLengths:
    communityDist[n] = communityDist.get(n, 0) + 1

  (X, Y) = zip(*[(n, communityDist[n] / len(label_to_nodeList)) for n in communityDist])

  (logX, logY) = ([np.log10(x) for x in X], [np.log10(y) for y in Y])

  resultLogLog = stats.linregress(logX, logY)
  return resultLogLog.slope*-1

def plotLFR():

    G1 = buildGraph()
    groundTruthNodes = set(getGroundTruthNodes())

    # Greedy Modularity
    greedyModularity = greedy_modularity_communities(G1)
    communityList=[]
    for node in G1.nodes():
        i=0
        for x in greedyModularity:
            if node in x and node in groundTruthNodes:                
                communityList.append(i)
                break
            i+=1
    label_to_nodeList={}
    a = set(communityList.values())
    for values in a:
        label_to_nodeList[values] = []
    for x in communityList:
        label_to_nodeList[communityList[x]].append(int(x))
    tau1 = getTau1(G1)
    tau2 = getTau2(label_to_nodeList)
    X = []
    Y = []
    for mu in np.arange(0.1, 1, 0.05):
        X.append(mu)
        G = LFR_benchmark_graph(29268, tau1, tau2, mu, average_degree=5, max_degree=50, min_community=20, max_community=100)
        communities = [(G.nodes[v]["community"]) for v in G]
        NMI = normalized_mutual_info_score(communities, communityList)
        Y.append(NMI)

    plt.plot(X, Y, '-o', label='Greedy Modularity', alpha=0.5)

    # Louvain
    partition = community.community_louvain.best_partition(G1)

    communityList=[]
    for nodes in G1.nodes():
        if nodes in groundTruthNodes:
            communityList.append(partition[nodes])
    
    X = []
    Y = []
    for mu in np.arange(0.1, 1, 0.05):
        X.append(mu)
        G = LFR_benchmark_graph(29268, tau1, tau2, mu, average_degree=5, max_degree=50, min_community=20, max_community=100)
        communities = [(G.nodes[v]["community"]) for v in G]
        NMI = normalized_mutual_info_score(communities, communityList)
        Y.append(NMI)

    plt.plot(X, Y, '-o', label='Louvain', alpha=0.5)
    plt.title('NMI vs Mixing Parameter')
    plt.xlabel('Mixing Parameter')
    plt.ylabel('Normalised Mutual Information')
    plt.legend(loc='upper right')
    plt.savefig('NMI_vs_mu_greedy.png')

def snowballsampling( G, seed, maxN):
    subgraph = set(seed)
    for x in seed:
        for e in nx.bfs_tree(G, source=x).nodes():
            if len(subgraph) < maxN:
                subgraph.add(e)
            else:
                return G.subgraph(subgraph)
    return G.subgraph(subgraph)

def randomWalkSample(G, seeds, steps):
    G_ = nx.Graph()
    for seed in seeds:
        source = seed
        for _ in range(steps):
            neigh = [node for node in G.neighbors(source)]
            if len(neigh) == 0:
                break
            target = np.random.choice(neigh, size=1)[0]
            G_.add_edge(source, target)
            source = target
    return G_
	 
def communitySampling(G: nx.Graph, sampleSize):
	randomNodes = set(random.sample(G.nodes(), 1))
	while len(randomNodes) < sampleSize:
		neighbors = [ neighbor
				for node in randomNodes
				for neighbor in G.neighbors(node) ]
		neighbors = list(set(neighbors).difference(randomNodes))
		random.shuffle(neighbors)
		expansion = 0
		for node in neighbors:
			newNode = random.choice(list(G.neighbors(node)))
			newExpansion = len(set(G.neighbors(node)).difference(randomNodes))
			if newExpansion >= expansion:
				expansion = newExpansion
				newNode = node
		randomNodes.add(newNode)
	return G.subgraph(randomNodes)

def comLens(communities):
	comLenList = []
	for i in communities:
		comLenList.append(len(i))
	return sorted(comLenList)

def getSampledGroundTruth(nodes):

	newComs = {}
	file = open('email-Eu-core-department-labels.txt', 'r')
	
	for line in file:
		com =  line.strip().split(" ")
		if int(com[0]) in nodes:
			newComs[int(com[0])]=int(com[1])

	allComs={}
	for key, val in newComs.items():
		allComs[val] = allComs.get(val, []) + [key]

	return allComs

def plotSamplingMethods():
	file1 = open('email-Eu-core.txt', 'r')
	Lines = file1.readlines()
	G = nx.Graph()
	for line in Lines:
		a, b = line.strip().split(" ")
		G.add_edge(int(a), int(b))
	
	largest_cc = max(nx.connected_components(G), key=len)
	Gcc = G.subgraph(largest_cc)

	mapping = {v:k for k, v in enumerate(list(Gcc.nodes))}

	Gnew = nx.Graph()
	file1 = open('email-Eu-core.txt', 'r')
	Lines = file1.readlines()
	for line in Gcc.edges:
		Gnew.add_edge(mapping.get(line[0]), mapping.get(line[1]))

	randomNode = random.choice(Gnew.nodes)

	G1 = randomWalkSample(Gnew, randomNode, 500)
	greedyModularity = greedy_modularity_communities(G1)
	plt.plot(comLens(greedyModularity), label = "Random Walk")
	plt.plot(comLens(getSampledGroundTruth(list(G1.nodes)).values()), label = "Ground Truth")
	plt.legend(loc='upper right')
	plt.xlabel('Number of communities')
	plt.ylabel('Community Size (sorted)')
	plt.savefig("Random", bbox_inches='tight')
	plt.clf()

	G1 = snowballsampling(Gnew, randomNode, 500)
	greedyModularity = greedy_modularity_communities(G1)
	plt.plot(comLens(greedyModularity), label = "Snowball")
	plt.plot(comLens(getSampledGroundTruth(list(G1.nodes)).values()), label = "Ground Truth")
	plt.legend(loc='upper right')
	plt.xlabel('Number of communities')
	plt.ylabel('Community Size (sorted)')
	plt.savefig("Snowball", bbox_inches='tight')
	plt.clf()

	
	G1 = communitySampling(Gnew, 500)
	
	greedyModularity = greedy_modularity_communities(G1)
	plt.plot(comLens(greedyModularity), label = "Expander")
	plt.plot(comLens(getSampledGroundTruth(list(G1.nodes)).values()), label = "Ground Truth")
	plt.legend(loc='upper right')
	plt.xlabel('Number of communities')
	plt.ylabel('Community Size (sorted)')
	plt.savefig("Expander", bbox_inches='tight')
	plt.clf()


if __name__ == '__main__':

	buildAdjList()
		
	if False:
		louvainCommDetNMI()
	if False:
		greedyCommDetNMI()
	if False:
		plotLFR()

	if False:
		G = buildGraph()
		greedyModularityCommunities(G)
		girvanNewmanCommunityDetection(G)
		louvainCommunities(G)

	if False:
		plotSamplingMethods()
            
