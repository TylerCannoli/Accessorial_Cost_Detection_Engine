#!/bin/bash
# Set up and run a local OSRM server with the US road network.
# Intended to run inside WSL Ubuntu with Docker.
#
# Prereqs: Docker running, us-latest.osm.pbf downloaded to ~/osrm/
# Outcome: OSRM server on http://localhost:5000
#
# Preprocessing memory: ~8-12 GB peak (US graph). Allow 30-60 min.
#
# Usage:
#   bash setup_osrm.sh prep    # extract + partition + customize
#   bash setup_osrm.sh serve   # start server on port 5000
#   bash setup_osrm.sh stop    # stop and remove container
#   bash setup_osrm.sh status  # show container status

set -e

OSRM_DIR="${OSRM_DIR:-$HOME/osrm}"
IMAGE="osrm/osrm-backend:latest"
CONTAINER="osrm-us"
PBF_NAME="us-latest"

cd "$OSRM_DIR"

case "${1:-serve}" in
  prep)
    if [ ! -f "${PBF_NAME}.osm.pbf" ]; then
      echo "ERROR: ${PBF_NAME}.osm.pbf not found in $OSRM_DIR"
      exit 1
    fi
    echo ">> Pulling $IMAGE (first run only)..."
    docker pull "$IMAGE"

    echo ">> Step 1/3: osrm-extract (~10-20 min, peak ~6 GB RAM)"
    docker run --rm -v "$OSRM_DIR":/data "$IMAGE" \
      osrm-extract -p /opt/car.lua "/data/${PBF_NAME}.osm.pbf"

    echo ">> Step 2/3: osrm-partition (~5-10 min)"
    docker run --rm -v "$OSRM_DIR":/data "$IMAGE" \
      osrm-partition "/data/${PBF_NAME}.osrm"

    echo ">> Step 3/3: osrm-customize (~5-10 min)"
    docker run --rm -v "$OSRM_DIR":/data "$IMAGE" \
      osrm-customize "/data/${PBF_NAME}.osrm"

    echo ">> Preprocessing complete. Run 'bash setup_osrm.sh serve' next."
    ;;

  serve)
    docker rm -f "$CONTAINER" 2>/dev/null || true
    docker run -d --name "$CONTAINER" \
      -p 5000:5000 \
      -v "$OSRM_DIR":/data \
      --restart unless-stopped \
      "$IMAGE" \
      osrm-routed --algorithm mld --max-table-size 10000 "/data/${PBF_NAME}.osrm"
    sleep 3
    echo ">> OSRM server running on http://localhost:5000"
    curl -s "http://localhost:5000/route/v1/driving/-90.34,38.74;-80.82,35.29?overview=false" | head -c 300
    echo ""
    ;;

  stop)
    docker rm -f "$CONTAINER" 2>/dev/null && echo "Stopped $CONTAINER" || echo "Not running"
    ;;

  status)
    docker ps -a --filter "name=$CONTAINER" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    ;;

  *)
    echo "Usage: bash setup_osrm.sh {prep|serve|stop|status}"
    exit 1
    ;;
esac
