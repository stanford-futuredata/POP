# POP Load Balancing

Requires Java 11 and an installation of CPLEX  v12.10.0!

To compile, run:

    mvn package

To test without POP:

    java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards SHARDS -numServers SERVERS -benchmark base

To test with POP:

    java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards SHARDS -numServers SERVERS -numSplits SPLITS -benchmark split

To test with the heuristic:

    java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards SHARDS -numServers SERVERS -benchmark heuristic