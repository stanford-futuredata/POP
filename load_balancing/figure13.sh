#! /bin/bash

echo "Unoptimized"
java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards 1024 -numServers 128 -benchmark base
echo "POP 4 Splits"
java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards 1024 -numServers 128 -numSplits 4 -benchmark split
echo "POP 16 Splits"
java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards 1024 -numServers 128 -numSplits 16 -benchmark split
echo "Heuristic"
java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards 1024 -numServers 128 -benchmark heuristic
