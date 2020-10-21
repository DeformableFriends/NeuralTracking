#include "cpu/graph_proc.h"

#include <set>
#include <vector>
#include <numeric> //std::iota
#include <algorithm>
#include <random> 
#include <iostream>

#include <Eigen/Dense>

using std::vector;

namespace graph_proc {

    py::array_t<bool> erode_mesh(const py::array_t<float>& vertexPositions, const py::array_t<int>& faceIndices, int nIterations, int minNeighbors) {
        int nVertices = vertexPositions.shape(0);
        int nFaces = faceIndices.shape(0);

        // Init output 
        py::array_t<bool> nonErodedVertices = py::array_t<bool>({ nVertices, 1 });
        std::vector<bool> nonErodedVerticesVec(nVertices, false);
        
        // Init list of eroded face indices with original list
        std::vector<Eigen::Vector3i> erodedFaceIndicesVec;
        erodedFaceIndicesVec.reserve(nFaces);
        for (int FaceIdx = 0; FaceIdx < nFaces; ++FaceIdx) {
            Eigen::Vector3i face(*faceIndices.data(FaceIdx, 0), *faceIndices.data(FaceIdx, 1), *faceIndices.data(FaceIdx, 2));
            erodedFaceIndicesVec.push_back(face);
        }

        // Erode mesh for a total of nIterations
        for (int i = 0; i < nIterations; i++) {
            nFaces = erodedFaceIndicesVec.size();

            // We compute the number of neighboring vertices for each vertex.
            vector<int> numNeighbors(nVertices, 0);
            for (int i = 0; i < nFaces; i++) {
                const auto& face = erodedFaceIndicesVec[i];
                numNeighbors[face[0]] += 1;
                numNeighbors[face[1]] += 1;
                numNeighbors[face[2]] += 1;
            }

            std::vector<Eigen::Vector3i> tmp;
            tmp.reserve(nFaces);

            for (int i = 0; i < nFaces; i++) {
                const auto& face = erodedFaceIndicesVec[i];
                if (numNeighbors[face[0]] >= minNeighbors && numNeighbors[face[1]] >= minNeighbors && numNeighbors[face[2]] >= minNeighbors) {
                    tmp.push_back(face);
                }
            }

            // We kill the faces with border vertices.
            erodedFaceIndicesVec.clear();
            erodedFaceIndicesVec = std::move(tmp);
        }

        // Mark non isolated vertices as not eroded.
        nFaces = erodedFaceIndicesVec.size();

        for (int i = 0; i < nFaces; i++) {
            const auto& face = erodedFaceIndicesVec[i];
            nonErodedVerticesVec[face[0]] = true;
            nonErodedVerticesVec[face[1]] = true;
            nonErodedVerticesVec[face[2]] = true;
        }

        // Store into python array
        for (int i = 0; i < nVertices; i++) {
            *nonErodedVertices.mutable_data(i, 0) = nonErodedVerticesVec[i];
        }

        return nonErodedVertices;
    }

    int sample_nodes(
        const py::array_t<float>& vertexPositions, const py::array_t<bool>& nonErodedVertices,
        py::array_t<float>& nodePositions, py::array_t<int>& nodeIndices, 
        float nodeCoverage, const bool useOnlyNonErodedIndices=true
    ) {
        // assert(vertexPositions.ndim() == 2);

        float nodeCoverage2 = nodeCoverage * nodeCoverage;
        int nVertices = vertexPositions.shape(0);
        // assert(vertexPositions.shape(1) == 3);
        // assert(nodePositions.shape(0) == nVertices);
        // assert(nodePositions.shape(1) == 3);
        // assert(nodeIndices.shape(0) == nVertices);
        // assert(nodeIndices.shape(1) == 1);

        nodePositions.resize({ nVertices, 3 }, false);
        nodeIndices.resize({ nVertices, 1 }, false);

        // create list of shuffled indices
        std::vector<int> shuffledVertices(nVertices);
        std::iota(std::begin(shuffledVertices), std::end(shuffledVertices), 0);

        std::default_random_engine re{std::random_device{}()};
        std::shuffle(std::begin(shuffledVertices), std::end(shuffledVertices), re);

        std::vector<Eigen::Vector3f> nodePositionsVec;
        for (int vertexIdx : shuffledVertices) {
        // for (int vertexIdx = 0; vertexIdx < nVertices; ++vertexIdx) {
            Eigen::Vector3f point(*vertexPositions.data(vertexIdx, 0), *vertexPositions.data(vertexIdx, 1), *vertexPositions.data(vertexIdx, 2));

            if (useOnlyNonErodedIndices && !(*nonErodedVertices.data(vertexIdx))) {
                continue;
            }

            bool bIsNode = true;
            for (int nodeIdx = 0; nodeIdx < nodePositionsVec.size(); ++nodeIdx) {
                if ((point - nodePositionsVec[nodeIdx]).squaredNorm() <= nodeCoverage2) {
                    bIsNode = false;
                    break;
                }
            }

            if (bIsNode) {
                nodePositionsVec.push_back(point);
                int newNodeIdx = nodePositionsVec.size() - 1;
                *nodePositions.mutable_data(newNodeIdx, 0) = point.x();
                *nodePositions.mutable_data(newNodeIdx, 1) = point.y();
                *nodePositions.mutable_data(newNodeIdx, 2) = point.z();
                *nodeIndices.mutable_data(newNodeIdx, 0) = vertexIdx;
            }
        }

        return nodePositionsVec.size();
    }

    /**
     * Custom comparison operator for geodesic priority queue.
     */
    struct CustomCompare {
        bool operator()(const std::pair<int, float>& left, const std::pair<int, float>& right) {
            return left.second > right.second;
        }
    };

    py::array_t<int> compute_edges_geodesic(
		const py::array_t<float>& vertexPositions,
		const py::array_t<int>& faceIndices, 
		const py::array_t<int>& nodeIndices, 
		int nMaxNeighbors, float maxInfluence
	) {
		int nVertices = vertexPositions.shape(0);
		int nFaces = faceIndices.shape(0);
        int nNodes = nodeIndices.shape(0);

        // Preprocess vertex neighbors.
		vector<std::set<int>> vertexNeighbors(nVertices);
        for (int faceIdx = 0; faceIdx < nFaces; faceIdx++) {
            for (int j = 0; j < 3; j++) {
                int v_idx = *faceIndices.data(faceIdx, j);
                
                for (int k = 0; k < 3; k++) {
                    int n_idx = *faceIndices.data(faceIdx, k);
                    
                    if (v_idx == n_idx) continue;
                    vertexNeighbors[v_idx].insert(n_idx);
                }
            }
        }

		// Compute inverse vertex -> node relationship.
		vector<int> mapVertexToNode(nVertices, -1);

		for (int nodeId = 0; nodeId < nNodes; nodeId++) {
			int vertexIdx = *nodeIndices.data(nodeId);
			if (vertexIdx >= 0) {
				mapVertexToNode[vertexIdx] = nodeId;
			}
		}

        // Compute node-vertex distances (not necessary).
        // py::array_t<float> nodeToVertexDistances = py::array_t<float>({ nNodes, nVertices });

		// for (int nodeId = 0; nodeId < nNodes; nodeId++) {
		// 	for (int vertexIdx = 0; vertexIdx < nVertices; vertexIdx++) {
		// 		*nodeToVertexDistances.mutable_data(nodeId, vertexIdx) = -1.f;
		// 	}
		// }

		// Construct geodesic edges.
        py::array_t<int> graphEdges = py::array_t<int>({ nNodes, nMaxNeighbors });

        // #pragma omp parallel for
		for (int nodeId = 0; nodeId < nNodes; nodeId++) {
			std::priority_queue<
				std::pair<int, float>,
				vector<std::pair<int, float>>,
				CustomCompare
			> nextVerticesWithIds;

			std::set<int> visitedVertices;

			// Add node vertex as the first vertex to be visited.
			int nodeVertexIdx = *nodeIndices.data(nodeId);
			if (nodeVertexIdx < 0) continue;
			nextVerticesWithIds.push(std::make_pair(nodeVertexIdx, 0.f));
			
			// Traverse all neighbors in the monotonically increasing order.
            vector<int> neighborNodeIds;
			while (!nextVerticesWithIds.empty()) {
				auto nextVertex = nextVerticesWithIds.top();
				nextVerticesWithIds.pop();

				int nextVertexIdx = nextVertex.first;
				float nextVertexDist = nextVertex.second;

				// We skip the vertex, if it was already visited before.
				if (visitedVertices.find(nextVertexIdx) != visitedVertices.end()) continue;

				// We check if the vertex is a node.
				int nextNodeId = mapVertexToNode[nextVertexIdx];
				if (nextNodeId >= 0 && nextNodeId != nodeId) {
                    neighborNodeIds.push_back(nextNodeId);
                    if (neighborNodeIds.size() >= nMaxNeighbors) break;
				}

				// Note down the node-vertex distance.
				// *nodeToVertexDistances.mutable_data(nodeId, nextVertexIdx) = nextVertexDist;

				// We visit the vertex, and check all his neighbors.
				// We add only vertices under a certain distance.
				visitedVertices.insert(nextVertexIdx);
				Eigen::Vector3f nextVertexPos(*vertexPositions.data(nextVertexIdx, 0), *vertexPositions.data(nextVertexIdx, 1), *vertexPositions.data(nextVertexIdx, 2));

				const auto& nextNeighbors = vertexNeighbors[nextVertexIdx];
				for (int neighborIdx : nextNeighbors) {
					Eigen::Vector3f  neighborVertexPos(*vertexPositions.data(neighborIdx, 0), *vertexPositions.data(neighborIdx, 1), *vertexPositions.data(neighborIdx, 2));
					float dist = nextVertexDist + (nextVertexPos - neighborVertexPos).norm();

					if (dist <= maxInfluence) {
						nextVerticesWithIds.push(std::make_pair(neighborIdx, dist));
					}
				}
			}

            // If we don't get any geodesic neighbors, we take one nearest Euclidean neighbor,
            // to have a constrained optimization system at non-rigid tracking.
            if (neighborNodeIds.empty()) {
                float nearestDistance2 = std::numeric_limits<float>::infinity();
                float nearestNodeId = -1;

                Eigen::Vector3f nodePos(*vertexPositions.data(nodeVertexIdx, 0), *vertexPositions.data(nodeVertexIdx, 1), *vertexPositions.data(nodeVertexIdx, 2));

                for (int i = 0; i < nNodes; i++) {
                    int vertexIdx = *nodeIndices.data(i);
                    if (i != nodeId && vertexIdx >= 0) {
                        Eigen::Vector3f neighborPos(*vertexPositions.data(vertexIdx, 0), *vertexPositions.data(vertexIdx, 1), *vertexPositions.data(vertexIdx, 2));

                        float distance2 = (neighborPos - nodePos).squaredNorm();
                        if (distance2 < nearestDistance2) {
                            nearestDistance2 = distance2;
                            nearestNodeId = i;
                        }
                    }
		        }

                if (nearestNodeId >= 0) neighborNodeIds.push_back(nearestNodeId);
            }

            // Store the nearest neighbors.
            int nNeighbors = neighborNodeIds.size();

            for (int i = 0; i < nNeighbors; i++) {
                *graphEdges.mutable_data(nodeId, i) = neighborNodeIds[i];
            }
            for (int i = nNeighbors; i < nMaxNeighbors; i++) {
                *graphEdges.mutable_data(nodeId, i) = -1;
            }
		}

        return graphEdges;
    }

    py::array_t<int> compute_edges_euclidean(const py::array_t<float>& nodePositions, int nMaxNeighbors) {
        int nNodes = nodePositions.shape(0);

        py::array_t<int> graphEdges = py::array_t<int>({ nNodes, nMaxNeighbors });

        // Find nearest Euclidean neighbors for each node.
        for (int nodeId = 0; nodeId < nNodes; nodeId++) {
            Eigen::Vector3f nodePos(*nodePositions.data(nodeId, 0), *nodePositions.data(nodeId, 1), *nodePositions.data(nodeId, 2));

            // Keep only the k nearest Euclidean neighbors.
            std::list<std::pair<int, float>> nearestNodesWithSquaredDistances;

            for (int neighborId = 0; neighborId < nNodes; neighborId++) {
                if (neighborId == nodeId) continue;

                Eigen::Vector3f neighborPos(*nodePositions.data(neighborId, 0), *nodePositions.data(neighborId, 1), *nodePositions.data(neighborId, 2));

                float distance2 = (nodePos - neighborPos).squaredNorm();
                bool bInserted = false;
                for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
                    // We insert the element at the first position where its distance is smaller than the other
                    // element's distance, which enables us to always keep a sorted list of at most k nearest
                    // neighbors.
                    if (distance2 <= it->second) {
                        it = nearestNodesWithSquaredDistances.insert(it, std::make_pair(neighborId, distance2));
                        bInserted = true;
                        break;
                    }
                }

                if (!bInserted && nearestNodesWithSquaredDistances.size() < nMaxNeighbors) {
                    nearestNodesWithSquaredDistances.emplace_back(std::make_pair(neighborId, distance2));
                }

                // We keep only the list of k nearest elements.
                if (bInserted && nearestNodesWithSquaredDistances.size() > nMaxNeighbors) {
                    nearestNodesWithSquaredDistances.pop_back();
                }
            }
            
            // Store nearest neighbor ids.
            int idx = 0;
            for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
                int neighborId = it->first;
                *graphEdges.mutable_data(nodeId, idx) = neighborId;
                idx++;
            }

            for (idx = nearestNodesWithSquaredDistances.size(); idx < nMaxNeighbors; idx++) {
                *graphEdges.mutable_data(nodeId, idx) = -1;
            }
        }

        return graphEdges;
    }

    static inline float computeAnchorWeight(const Eigen::Vector3f& pointPosition, const Eigen::Vector3f& nodePosition, float nodeCoverage) {
        return std::exp(-(nodePosition - pointPosition).squaredNorm() / (2.f * nodeCoverage * nodeCoverage));
    }

    void compute_pixel_anchors_geodesic(
        const py::array_t<float>& graphNodes, 
        const py::array_t<int>& graphEdges,
        const py::array_t<float>& pointImage,
        int neighborhoodDepth,
        float nodeCoverage,
        py::array_t<int>& pixelAnchors, 
        py::array_t<float>& pixelWeights
    ) {
        int numNodes = graphNodes.shape(0);
        int numNeighbors = graphEdges.shape(1);
        int width = pointImage.shape(2);
        int height = pointImage.shape(1);
        // int nChannels = pointImage.shape(0);

        // Allocate graph node ids and corresponding skinning weights.
        // Initialize with invalid anchors.
        pixelAnchors.resize({ height, width, GRAPH_K }, false);
        pixelWeights.resize({ height, width, GRAPH_K }, false);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int k = 0; k < GRAPH_K; k++) {
                    *pixelAnchors.mutable_data(y, x, k) = -1;
                    *pixelWeights.mutable_data(y, x, k) = 0.f;
                }
            }
        }

        // Compute anchors for every pixel.
        #pragma omp parallel for
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Query 3d pixel position.
                Eigen::Vector3f pixelPos(*pointImage.data(0, y, x), *pointImage.data(1, y, x), *pointImage.data(2, y, x));
                if (pixelPos.z() <= 0) continue;
                
                // Find nearest Euclidean graph node.
                float nearestDist2 = std::numeric_limits<float>::infinity();
                int nearestNodeId = -1;
                for (int nodeId = 0; nodeId < numNodes; nodeId++) {
                    Eigen::Vector3f nodePos(*graphNodes.data(nodeId, 0), *graphNodes.data(nodeId, 1), *graphNodes.data(nodeId, 2));
                    float dist2 = (nodePos - pixelPos).squaredNorm();
                    if (dist2 < nearestDist2) {
                        nearestDist2 = dist2;
                        nearestNodeId = nodeId;
                    }
                }

                // Compute the geodesic neighbor candidates.
                std::set<int> neighbors{ nearestNodeId };
                std::set<int> newNeighbors = neighbors;

                for (int i = 0; i < neighborhoodDepth; i++) {
                    // Get all neighbors of the new neighbors.
                    std::set<int> currentNeighbors;
                    for (auto neighborId : newNeighbors) {
                        for (int k = 0; k < numNeighbors; k++) {
                            int currentNeighborId = *graphEdges.data(neighborId, k);
                            
                            if (currentNeighborId >= 0) {
                                currentNeighbors.insert(currentNeighborId);
                            }
                        }
                    }

                    // Check the newly added neighbors (not present in the neighbors set).
                    newNeighbors.clear();
                    std::set_difference(
                        currentNeighbors.begin(), currentNeighbors.end(),
                        neighbors.begin(), neighbors.end(),
                        std::inserter(newNeighbors, newNeighbors.begin())
                    );

                    // Insert the newly added neighbors.
                    neighbors.insert(newNeighbors.begin(), newNeighbors.end());
                }

                // Keep only the k nearest geodesic neighbors.
                std::list<std::pair<int, float>> nearestNodesWithSquaredDistances;

                for (auto&& neighborId : neighbors) {
                    Eigen::Vector3f nodePos(*graphNodes.data(neighborId, 0), *graphNodes.data(neighborId, 1), *graphNodes.data(neighborId, 2));

                    float distance2 = (nodePos - pixelPos).squaredNorm();
                    bool bInserted = false;
                    for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
                        // We insert the element at the first position where its distance is smaller than the other
                        // element's distance, which enables us to always keep a sorted list of at most k nearest
                        // neighbors.
                        if (distance2 <= it->second) {
                            it = nearestNodesWithSquaredDistances.insert(it, std::make_pair(neighborId, distance2));
                            bInserted = true;
                            break;
                        }
                    }

                    if (!bInserted && nearestNodesWithSquaredDistances.size() < GRAPH_K) {
                        nearestNodesWithSquaredDistances.emplace_back(std::make_pair(neighborId, distance2));
                    }

                    // We keep only the list of k nearest elements.
                    if (bInserted && nearestNodesWithSquaredDistances.size() > GRAPH_K) {
                        nearestNodesWithSquaredDistances.pop_back();
                    }
                }

                // Compute skinning weights.
                std::vector<int> nearestGeodesicNodeIds;
                nearestGeodesicNodeIds.reserve(nearestNodesWithSquaredDistances.size());

                std::vector<float> skinningWeights;
                skinningWeights.reserve(nearestNodesWithSquaredDistances.size());

                float weightSum{ 0.f };
                for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
                    int nodeId = it->first;

                    Eigen::Vector3f nodePos(*graphNodes.data(nodeId, 0), *graphNodes.data(nodeId, 1), *graphNodes.data(nodeId, 2));
                    float weight = computeAnchorWeight(pixelPos, nodePos, nodeCoverage);
                    weightSum += weight;

                    nearestGeodesicNodeIds.push_back(nodeId);
                    skinningWeights.push_back(weight);
                }

                // Normalize the skinning weights.
                int nAnchors = nearestGeodesicNodeIds.size();

                if (weightSum > 0) {
                    for (int i = 0; i < nAnchors; i++)	skinningWeights[i] /= weightSum;
                }
                else if (nAnchors > 0) {
                    for (int i = 0; i < nAnchors; i++)	skinningWeights[i] = 1.f / nAnchors;
                }
                
                // Store the results.
                for (int i = 0; i < nAnchors; i++) {
                    *pixelAnchors.mutable_data(y, x, i) = nearestGeodesicNodeIds[i];
                    *pixelWeights.mutable_data(y, x, i) = skinningWeights[i];
                }
            }
        }
    }

    void compute_pixel_anchors_euclidean(
        const py::array_t<float>& graphNodes, 
        const py::array_t<float>& pointImage,
        float nodeCoverage,
        py::array_t<int>& pixelAnchors, 
        py::array_t<float>& pixelWeights
    ) {
        int nNodes = graphNodes.shape(0);
        int width = pointImage.shape(2);
        int height = pointImage.shape(1);
        // int nChannels = pointImage.shape(0);

        // Allocate graph node ids and corresponding skinning weights.
        // Initialize with invalid anchors.
        pixelAnchors.resize({ height, width, GRAPH_K }, false);
        pixelWeights.resize({ height, width, GRAPH_K }, false);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int k = 0; k < GRAPH_K; k++) {
                    *pixelAnchors.mutable_data(y, x, k) = -1;
                    *pixelWeights.mutable_data(y, x, k) = 0.f;
                }
            }
        }

        // Compute anchors for every pixel.
        #pragma omp parallel for
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Query 3d pixel position.
                Eigen::Vector3f pixelPos(*pointImage.data(0, y, x), *pointImage.data(1, y, x), *pointImage.data(2, y, x));
                if (pixelPos.z() <= 0) continue;
                
                // Keep only the k nearest Euclidean neighbors.
                std::list<std::pair<int, float>> nearestNodesWithSquaredDistances;

                for (int nodeId = 0; nodeId < nNodes; nodeId++) {
                    Eigen::Vector3f nodePos(*graphNodes.data(nodeId, 0), *graphNodes.data(nodeId, 1), *graphNodes.data(nodeId, 2));

                    float distance2 = (pixelPos - nodePos).squaredNorm();
                    bool bInserted = false;
                    for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
                        // We insert the element at the first position where its distance is smaller than the other
                        // element's distance, which enables us to always keep a sorted list of at most k nearest
                        // neighbors.
                        if (distance2 <= it->second) {
                            it = nearestNodesWithSquaredDistances.insert(it, std::make_pair(nodeId, distance2));
                            bInserted = true;
                            break;
                        }
                    }

                    if (!bInserted && nearestNodesWithSquaredDistances.size() < GRAPH_K) {
                        nearestNodesWithSquaredDistances.emplace_back(std::make_pair(nodeId, distance2));
                    }

                    // We keep only the list of k nearest elements.
                    if (bInserted && nearestNodesWithSquaredDistances.size() > GRAPH_K) {
                        nearestNodesWithSquaredDistances.pop_back();
                    }
                }

                // Compute skinning weights.
                std::vector<int> nearestEuclideanNodeIds;
                nearestEuclideanNodeIds.reserve(nearestNodesWithSquaredDistances.size());

                std::vector<float> skinningWeights;
                skinningWeights.reserve(nearestNodesWithSquaredDistances.size());

                float weightSum{ 0.f };
                for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
                    int nodeId = it->first;

                    Eigen::Vector3f nodePos(*graphNodes.data(nodeId, 0), *graphNodes.data(nodeId, 1), *graphNodes.data(nodeId, 2));
                    float weight = computeAnchorWeight(pixelPos, nodePos, nodeCoverage);
                    weightSum += weight;

                    nearestEuclideanNodeIds.push_back(nodeId);
                    skinningWeights.push_back(weight);
                }

                // Normalize the skinning weights.
                int nAnchors = nearestEuclideanNodeIds.size();

                if (weightSum > 0) {
                    for (int i = 0; i < nAnchors; i++)	skinningWeights[i] /= weightSum;
                }
                else if (nAnchors > 0) {
                    for (int i = 0; i < nAnchors; i++)	skinningWeights[i] = 1.f / nAnchors;
                }
                
                // Store the results.
                for (int i = 0; i < nAnchors; i++) {
                    *pixelAnchors.mutable_data(y, x, i) = nearestEuclideanNodeIds[i];
                    *pixelWeights.mutable_data(y, x, i) = skinningWeights[i];
                }
            }
        }
    }

    void construct_regular_graph(
        const py::array_t<float>& pointImage,
        int xNodes, int yNodes,
        float edgeThreshold,
        float maxPointToNodeDistance,
        float maxDepth,
        py::array_t<float>& graphNodes, 
        py::array_t<int>& graphEdges, 
        py::array_t<int>& pixelAnchors, 
        py::array_t<float>& pixelWeights
    ) {
        int width = pointImage.shape(2);
        int height = pointImage.shape(1);
        int nChannels = pointImage.shape(0);

        float xStep = float(width - 1) / (xNodes - 1);
        float yStep = float(height - 1) / (yNodes - 1);

        // Sample graph nodes.
        // We need to maintain the mapping from all -> valid nodes ids.
        int nNodes = xNodes * yNodes;
        std::vector<int> sampledNodeMapping(nNodes, -1);

        std::vector<Eigen::Vector3f> nodePositions;
        nodePositions.reserve(nNodes);

        int nodeId = 0;
        for (int y = 0; y < yNodes; y++) {
            for (int x = 0; x < xNodes; x++) {
                int nodeIdx = y * xNodes + x;

                // We use nearest neighbor interpolation for node position
                // computation.
                int xPixel = std::round(x * xStep); 
                int yPixel = std::round(y * yStep);

                Eigen::Vector3f pixelPos(*pointImage.data(0, yPixel, xPixel), *pointImage.data(1, yPixel, xPixel), *pointImage.data(2, yPixel, xPixel));
                if (pixelPos.z() <= 0 || pixelPos.z() > maxDepth) continue;

                nodePositions.push_back(pixelPos);
                sampledNodeMapping[nodeIdx] = nodeId;
                nodeId++;
            }
        }
        int nSampledNodes = nodeId;

        // Compute graph edges using pixel-wise connectivity. Each node
        // is connected with at most 8 neighboring pixels.
        int numNeighbors = 8;
        float edgeThreshold2 = edgeThreshold * edgeThreshold;

        std::vector<int> sampledNodeEdges(nSampledNodes * numNeighbors, -1);
        std::vector<bool> connectedNodes(nSampledNodes, false);

        int nConnectedNodes = 0;
        for (int y = 0; y < yNodes; y++) {
            for (int x = 0; x < xNodes; x++) {
                int nodeIdx = y * xNodes + x;
                int nodeId = sampledNodeMapping[nodeIdx];

                if (nodeId >= 0) {
                    Eigen::Vector3f nodePosition = nodePositions[nodeId];

                    int neighborCount = 0;
                    for (int yDelta = -1; yDelta <= 1; yDelta++) {
                        for (int xDelta = -1; xDelta <= 1; xDelta++) {
                            int xNeighbor = x + xDelta;
                            int yNeighbor = y + yDelta;
                            if (xNeighbor < 0 || xNeighbor >= xNodes || yNeighbor < 0 || yNeighbor >= yNodes)
                                continue;
                            
                            int neighborIdx = yNeighbor * xNodes + xNeighbor;
                            
                            if (neighborIdx == nodeIdx || neighborIdx < 0)
                                continue;

                            int neighborId = sampledNodeMapping[neighborIdx];
                            if (neighborId >= 0) {
                                Eigen::Vector3f neighborPosition = nodePositions[neighborId];

                                if ((neighborPosition - nodePosition).squaredNorm() <= edgeThreshold2) {
                                    sampledNodeEdges[nodeId * numNeighbors + neighborCount] = neighborId;
                                    neighborCount++;
                                }
                            }
                        }
                    }

                    for (int i = neighborCount; i < numNeighbors; i++) {
                        sampledNodeEdges[nodeId * numNeighbors + i] = -1;
                    }

                    if (neighborCount > 0) {
                        connectedNodes[nodeId] = true;
                        nConnectedNodes += 1;
                    }
                }
            }
        }

        // Filter out nodes with no edges.
        // After changing node ids the edge ids need to be changed as well.
        std::vector<int> validNodeMapping(nSampledNodes, -1);

        graphNodes.resize({ nConnectedNodes, 3 }, false);
        graphEdges.resize({ nConnectedNodes, numNeighbors }, false);

        int validNodeId = 0;
        for (int y = 0; y < yNodes; y++) {
            for (int x = 0; x < xNodes; x++) {
                int nodeIdx = y * xNodes + x;
                int nodeId = sampledNodeMapping[nodeIdx];

                if (nodeId >= 0 && connectedNodes[nodeId]) {
                    validNodeMapping[nodeId] = validNodeId;

                    Eigen::Vector3f nodePosition = nodePositions[nodeId];
                    *graphNodes.mutable_data(validNodeId, 0) = nodePosition.x();
                    *graphNodes.mutable_data(validNodeId, 1) = nodePosition.y();
                    *graphNodes.mutable_data(validNodeId, 2) = nodePosition.z();

                    validNodeId++;
                }
            }
        }

        for (int y = 0; y < yNodes; y++) {
            for (int x = 0; x < xNodes; x++) {
                int nodeIdx = y * xNodes + x;
                int nodeId = sampledNodeMapping[nodeIdx];

                if (nodeId >= 0 && connectedNodes[nodeId]) {
                    int validNodeId = validNodeMapping[nodeId];
                    
                    if (validNodeId >= 0) {
                        for (int i = 0; i < numNeighbors; i++) {
                            int sampledNeighborId = sampledNodeEdges[nodeId * numNeighbors + i];
                            if (sampledNeighborId >= 0) {
                                *graphEdges.mutable_data(validNodeId, i) = validNodeMapping[sampledNeighborId];
                            }
                            else {
                                *graphEdges.mutable_data(validNodeId, i) = -1;
                            }
                        }
                    }
                }
            }
        }

        // Compute pixel anchors and weights.
        pixelAnchors.resize({ height, width, 4 }, false);
        pixelWeights.resize({ height, width, 4 }, false);

        float maxPointToNodeDistance2 = maxPointToNodeDistance * maxPointToNodeDistance;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Initialize with invalid values. 
                for (int k = 0; k < 4; k++) {
                    *pixelAnchors.mutable_data(y, x, k) = -1;
                    *pixelWeights.mutable_data(y, x, k) = 0.f;
                }
                
                // Compute 4 nearest nodes.
                float xNode = float(x) / xStep;
                float yNode = float(y) / yStep;

                int x0 = std::floor(xNode), x1 = x0 + 1;
                int y0 = std::floor(yNode), y1 = y0 + 1;

                // Check that all neighboring nodes are valid.
                if (x0 < 0 || x1 >= xNodes || y0 < 0 || y1 >= yNodes)
                    continue;

                int sampledNode00 = sampledNodeMapping[y0 * xNodes + x0];
                int sampledNode01 = sampledNodeMapping[y1 * xNodes + x0];
                int sampledNode10 = sampledNodeMapping[y0 * xNodes + x1];
                int sampledNode11 = sampledNodeMapping[y1 * xNodes + x1];

                if (sampledNode00 < 0 || sampledNode01 < 0 || sampledNode10 < 0 || sampledNode11 < 0)
                    continue;

                int validNode00 = validNodeMapping[sampledNode00];
                int validNode01 = validNodeMapping[sampledNode01];
                int validNode10 = validNodeMapping[sampledNode10];
                int validNode11 = validNodeMapping[sampledNode11];

                if (validNode00 < 0 || validNode01 < 0 || validNode10 < 0 || validNode11 < 0)
                    continue;

                // Check that all nodes are close enough to the point.
                Eigen::Vector3f pixelPos(*pointImage.data(0, y, x), *pointImage.data(1, y, x), *pointImage.data(2, y, x));
                if (pixelPos.z() <= 0 || pixelPos.z() > maxDepth) continue;

                if ((pixelPos - nodePositions[sampledNode00]).squaredNorm() > maxPointToNodeDistance2 ||
                    (pixelPos - nodePositions[sampledNode01]).squaredNorm() > maxPointToNodeDistance2 ||
                    (pixelPos - nodePositions[sampledNode10]).squaredNorm() > maxPointToNodeDistance2 ||
                    (pixelPos - nodePositions[sampledNode11]).squaredNorm() > maxPointToNodeDistance2
                ) {
                    continue;
                }

                // Compute bilinear weights.
                float dx = xNode - x0;
                float dy = yNode - y0;

                float w00 = (1 - dx) * (1 - dy);
                float w01 = (1 - dx) * dy;
                float w10 = dx * (1 - dy);
                float w11 = dx * dy;
                
                *pixelAnchors.mutable_data(y, x, 0) = validNode00;
                *pixelWeights.mutable_data(y, x, 0) = w00;
                *pixelAnchors.mutable_data(y, x, 1) = validNode01;
                *pixelWeights.mutable_data(y, x, 1) = w01;
                *pixelAnchors.mutable_data(y, x, 2) = validNode10;
                *pixelWeights.mutable_data(y, x, 2) = w10;
                *pixelAnchors.mutable_data(y, x, 3) = validNode11;
                *pixelWeights.mutable_data(y, x, 3) = w11;
            }
        }
    }

} // namespace graph_proc