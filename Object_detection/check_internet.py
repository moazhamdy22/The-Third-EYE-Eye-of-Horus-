import requests
import speedtest
import socket
import time
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# Silence speedtest-cli's internal logging if desired
logging.getLogger('speedtest').setLevel(logging.WARNING)

def check_connectivity(host="www.google.com", port=80, timeout=5):
    """Checks basic internet connectivity by trying to connect to a host."""
    logger.info(f"Checking connectivity to {host}:{port}...")
    try:
        # Try resolving the host first (checks DNS)
        socket.gethostbyname(host)
        # Then try establishing a basic TCP connection
        socket.create_connection((host, port), timeout=timeout)
        logger.info("Connectivity check successful.")
        return True
    except socket.gaierror:
        logger.error(f"DNS lookup failed for {host}.")
        return False
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        logger.error(f"Connection to {host}:{port} failed: {e}")
        return False

def measure_speed():
    """Measures download and upload speed using speedtest-cli."""
    logger.info("Initializing speed test...")
    try:
        st = speedtest.Speedtest()

        # --- Optional: Find best server (can take time) ---
        # logger.info("Finding optimal server...")
        # st.get_best_server()
        # server_info = st.results.server
        # logger.info(f"Using server: {server_info['sponsor']} ({server_info['name']}, {server_info['country']})")
        # --- End Optional ---

        logger.info("Measuring download speed...")
        download_speed_bps = st.download() # Returns bits per second
        download_speed_mbps = download_speed_bps / 1_000_000 # Convert to Megabits per second

        logger.info("Measuring upload speed...")
        upload_speed_bps = st.upload() # Returns bits per second
        upload_speed_mbps = upload_speed_bps / 1_000_000 # Convert to Megabits per second

        # Also get ping (latency)
        ping_ms = st.results.ping

        return download_speed_mbps, upload_speed_mbps, ping_ms

    except speedtest.SpeedtestException as e:
        logger.error(f"Speed test failed: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during speed test: {e}", exc_info=True)
        return None, None, None

if __name__ == "__main__":
    print("-" * 30)
    print("Internet Connectivity & Speed Test")
    print("-" * 30)

    # 1. Check basic connectivity
    if not check_connectivity():
        print("\nRESULT: Basic internet connectivity FAILED.")
        print("Cannot perform speed test.")
        exit()

    # 2. Measure Speed
    print("\nStarting speed test (this may take a minute)...")
    download_mbps, upload_mbps, ping_ms = measure_speed()

    # 3. Display Results
    print("\n--- Test Results ---")
    if download_mbps is not None and upload_mbps is not None:
        print(f"Ping (Latency): {ping_ms:.2f} ms")
        print(f"Download Speed: {download_mbps:.2f} Mbps")
        print(f"Upload Speed:   {upload_mbps:.2f} Mbps")

        # Simple check relevant to video upload
        min_upload_for_video = 5.0 # Mbps - Adjust as needed (HD might need more)
        if upload_mbps >= min_upload_for_video:
            print(f"\nUpload speed ({upload_mbps:.2f} Mbps) looks potentially sufficient for video uploads (>{min_upload_for_video} Mbps).")
        else:
            print(f"\nWARNING: Upload speed ({upload_mbps:.2f} Mbps) might be slow for large video uploads (recommend >{min_upload_for_video} Mbps).")

    else:
        print("Speed test could not be completed.")

    print("-" * 30)