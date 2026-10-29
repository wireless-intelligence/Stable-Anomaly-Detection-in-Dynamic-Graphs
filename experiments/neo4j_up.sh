#!/bin/bash
set -e

NAME=${NAME:-sag-neo4j}
USER=${NEO4J_USER:-neo4j}
PASSWORD=${NEO4J_PASSWORD:-changeme}

if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Container ${NAME} already running"
else
  if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "Starting existing ${NAME}"
    docker start "${NAME}"
  else
    echo "Creating ${NAME}"
    docker run -d \
      --name "${NAME}" \
      -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH="${USER}/${PASSWORD}" \
      -e NEO4J_dbms_memory_heap_max__size=2G \
      -e NEO4J_dbms_memory_pagecache_size=1G \
      neo4j:5
  fi
fi

echo
echo "Neo4j starting. It typically takes ~15s to be ready."
echo "Browser UI : http://localhost:7474"
echo "Bolt       : bolt://127.0.0.1:7687"
echo "User       : ${USER}"
echo "Password   : ${PASSWORD}"
echo
echo "Export before running the benchmark:"
echo "  export NEO4J_URI=bolt://127.0.0.1:7687"
echo "  export NEO4J_USER=${USER}"
echo "  export NEO4J_PASSWORD=${PASSWORD}"
