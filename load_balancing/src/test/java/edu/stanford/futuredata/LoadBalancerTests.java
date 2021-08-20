package edu.stanford.futuredata;


import ilog.concert.IloException;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class LoadBalancerTests {
    private static final Logger logger = LoggerFactory.getLogger(LoadBalancerTests.class);

    @Test
    public void testBalanceLoadFunction() throws IloException {
        logger.info("testBalanceLoadFunction");

        int numShards = 4;
        int numServers = 2;
        int[] shardLoads = new int[]{1, 2, 3, 20};
        int[] memoryUsages = new int[]{9, 1, 1, 1};
        int[][] currentLocations = new int[][]{new int[]{1, 1, 1, 1}, new int[]{0, 0, 0, 0}};
        int maxMemory = 10;

        List<double[]> returnR = new LoadBalancer().balanceLoad(numShards, numServers, shardLoads, memoryUsages, currentLocations, new HashMap<>(), maxMemory);
        logger.info("{} {}", returnR.get(0), returnR.get(1));
        double averageLoad = IntStream.of(shardLoads).sum() / (double) numServers;
        for(double[] Rs: returnR) {
            double serverLoad = 0;
            for(int i = 0; i < numShards; i++) {
                serverLoad += Rs[i] * shardLoads[i];
            }
            assertTrue(serverLoad <= averageLoad * 1.05);
        }
    }

    @Test
    public void testBalanceLoadHeuristic() {
        logger.info("testBalanceLoadHeuristic");

        Map<Integer, Integer> shardLoads = Map.of(0, 5, 1, 5, 2, 5, 3, 15);
        Map<Integer, Integer> shardMap = Map.of(0, 0, 1, 0, 2, 0, 3, 0);
        List<Integer> serverList = List.of(0, 1);

        Map<Integer, Integer> result = LoadBalancer.heuristicBalance(shardLoads, shardMap, serverList);
        assertNotEquals(result.get(0), result.get(3));
        assertEquals(result.get(0), result.get(1));
        assertEquals(result.get(0), result.get(2));
    }
}
