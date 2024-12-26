from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel, info
import os
import time
import random
import matplotlib.pyplot as plt
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
from collections import deque
from datetime import datetime
import numpy as np

class RoundRobinLoadBalancer:
    def _init_(self, server_ips):
        self.servers = deque(server_ips)
        self.metrics = {
            'response_times': [],
            'throughputs': [],
            'utilizations': [],
            'requests_per_server': {ip: 0 for ip in server_ips},
            'timestamps': [],
            'channel_capacity': 10  # Mbps
        }
    
    def get_next_server(self):
        self.servers.rotate(-1)
        return self.servers[0]

    def update_metrics(self, server_ip, response_time, bytes_transferred, timestamp):
        self.metrics['response_times'].append(response_time)
        self.metrics['requests_per_server'][server_ip] += 1
        
        # Calculate instantaneous throughput for this request (Mbps)
        duration = max(response_time / 1000, 0.001)  # Convert ms to seconds, minimum 1ms
        throughput = (bytes_transferred * 8) / (duration * 1000000)
        self.metrics['throughputs'].append(throughput)
        
        # Calculate channel utilization for this request
        utilization = (throughput / self.metrics['channel_capacity']) * 100
        self.metrics['utilizations'].append(utilization)
        self.metrics['timestamps'].append(timestamp)

def simulate_traffic(client, load_balancer, num_requests=100):
    """
    Simulate traffic using round-robin load balancing and collect metrics
    """
    start_time = time.time()
    
    # Create a small test file on each server
    for server_ip in load_balancer.servers:
        client.cmd(f'echo "test data" | ssh {server_ip} "cat > /tmp/testfile"')
    
    for i in range(num_requests):
        server_ip = load_balancer.get_next_server()
        info(f"\rProcessing request {i+1}/{num_requests} ")
        
        request_start = time.time()
        # Use wget instead of curl and fetch a known file
        result = client.cmd(f"wget -q -O - http://{server_ip}/tmp/testfile")
        request_end = time.time()
        
        bytes_transferred = len(result)
        response_time = (request_end - request_start) * 1000  # Convert to ms
        timestamp = request_end - start_time
        
        load_balancer.update_metrics(server_ip, response_time, bytes_transferred, timestamp)
        time.sleep(0.1)  # Small delay between requests
    
    info("\nTraffic simulation completed\n")

def visualize_metrics(metrics):
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Server Load Distribution
        servers = list(metrics['requests_per_server'].keys())
        requests = list(metrics['requests_per_server'].values())
        ax1.bar(servers, requests)
        ax1.set_title('Load Distribution Across Servers')
        ax1.set_xlabel('Server IP')
        ax1.set_ylabel('Number of Requests')
        
        # Plot 2: Response Times Over Time
        ax2.plot(metrics['timestamps'], metrics['response_times'], 'b-')
        ax2.set_title('Response Times Over Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Response Time (ms)')
        ax2.grid(True)
        
        # Plot 3: Throughput Over Time
        ax3.plot(metrics['timestamps'], metrics['throughputs'], 'g-')
        ax3.set_title('Network Throughput Over Time')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Throughput (Mbps)')
        ax3.grid(True)
        
        # Plot 4: Channel Utilization Over Time
        ax4.plot(metrics['timestamps'], metrics['utilizations'], 'r-')
        ax4.set_title('Channel Utilization Over Time')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Utilization (%)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        info(f"Error creating visualizations: {str(e)}\n")

def simulate_network():
    """
    Create and simulate network with round-robin load balancing
    """
    net = None
    try:
        net = Mininet(controller=Controller, link=TCLink, switch=OVSSwitch)

        info("* Creating network components\n")
        c0 = net.addController('c0')
        
        # Add hosts
        h1 = net.addHost('h1', ip='10.0.0.1')
        h2 = net.addHost('h2', ip='10.0.0.2')
        h3 = net.addHost('h3', ip='10.0.0.3')
        h4 = net.addHost('h4', ip='10.0.0.4')
        h5 = net.addHost('h5', ip='10.0.0.5')
        h6 = net.addHost('h6', ip='10.0.0.6')
        
        s1 = net.addSwitch('s1')

        # Create links with 10 Mbps bandwidth and 5ms delay
        for h in [h1, h2, h3, h4, h5, h6]:
            net.addLink(h, s1, bw=10, delay='5ms')

        info("* Starting network\n")
        net.build()
        c0.start()
        s1.start([c0])

        # Start a simple HTTP server on each backend host
        info("* Starting backend servers\n")
        for h in [h2, h3, h4, h5, h6]:
            h.cmd('mkdir -p /tmp')
            h.cmd('python3 -m http.server 80 &> /dev/null &')

        # Initialize load balancer and run simulation
        server_ips = ['10.0.0.2', '10.0.0.3', '10.0.0.4', '10.0.0.5', '10.0.0.6']
        load_balancer = RoundRobinLoadBalancer(server_ips)

        info("* Running traffic simulation\n")
        simulate_traffic(h1, load_balancer, num_requests=100)

        info("* Generating visualization\n")
        visualize_metrics(load_balancer.metrics)

    except Exception as e:
        info(f"Error in simulation: {str(e)}\n")
    
    finally:
        if net:
            info("* Cleaning up network\n")
            net.stop()

if _name_ == '_main_':
    setLogLevel('info')
    simulate_network()
