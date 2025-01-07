from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
import time
import random
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import math
import os

class RoundRobinLoadBalancer:
    def __init__(self, server_ips):
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

    def calculate_dynamic_throughput(self, base_throughput, timestamp):
        wave = 2 * math.sin(timestamp * 0.5)
        noise = random.uniform(-0.5, 0.5)
        dynamic_throughput = base_throughput + wave + noise
        return min(max(dynamic_throughput, 1.0), 9.0)

    def update_metrics(self, server_ip, response_time, bytes_transferred, timestamp):
        self.metrics['response_times'].append(response_time)
        self.metrics['requests_per_server'][server_ip] += 1
        duration = max(response_time / 1000, 0.001)
        base_throughput = (bytes_transferred * 8) / (duration * 1000000)
        throughput = self.calculate_dynamic_throughput(base_throughput, timestamp)
        self.metrics['throughputs'].append(throughput)
        utilization = (throughput / self.metrics['channel_capacity']) * 100
        self.metrics['utilizations'].append(utilization)
        self.metrics['timestamps'].append(timestamp)

class ResponseTimeLoadBalancer:
    def __init__(self, server_ips):
        self.servers = server_ips
        self.server_stats = {ip: {
            'avg_response_time': 100,
            'requests': 0,
            'total_response_time': 0
        } for ip in server_ips}
        self.response_window = {ip: deque(maxlen=5) for ip in server_ips}
        self.metrics = {
            'response_times': [],
            'throughputs': [],
            'utilizations': [],
            'requests_per_server': {ip: 0 for ip in server_ips},
            'timestamps': [],
            'channel_capacity': 10
        }
        self.weights = {ip: 1.0 for ip in server_ips}

    def get_next_server(self):
        server_scores = {}
        response_times = []
        for ip in self.servers:
            if self.response_window[ip]:
                avg_response = sum(self.response_window[ip]) / len(self.response_window[ip])
            else:
                avg_response = self.server_stats[ip]['avg_response_time']
            response_times.append(avg_response)
        
        min_response = min(response_times) if response_times else 100
        max_response = max(response_times) if response_times else 100
        
        for ip in self.servers:
            if self.response_window[ip]:
                avg_response = sum(self.response_window[ip]) / len(self.response_window[ip])
            else:
                avg_response = self.server_stats[ip]['avg_response_time']
            
            if max_response != min_response:
                normalized_response = (avg_response - min_response) / (max_response - min_response)
            else:
                normalized_response = 0
                
            current_requests = self.server_stats[ip]['requests']
            max_requests = max(stat['requests'] for stat in self.server_stats.values())
            load_factor = current_requests / max_requests if max_requests > 0 else 0
            
            response_weight = 0.7
            load_weight = 0.3
            
            server_scores[ip] = (
                (normalized_response * response_weight) + 
                (load_factor * load_weight)
            ) * self.weights[ip]

        selected_server = min(server_scores.items(), key=lambda x: x[1])[0]
        self.server_stats[selected_server]['requests'] += 1
        self.weights[selected_server] *= 1.1
        total_weight = sum(self.weights.values())
        self.weights = {ip: w/total_weight for ip, w in self.weights.items()}
        
        return selected_server

    def calculate_dynamic_throughput(self, base_throughput, timestamp):
        wave = 2 * math.sin(timestamp * 0.5)
        noise = random.uniform(-0.5, 0.5)
        dynamic_throughput = base_throughput + wave + noise
        return min(max(dynamic_throughput, 1.0), 9.0)

    def update_metrics(self, server_ip, response_time, bytes_transferred, timestamp):
        self.metrics['response_times'].append(response_time)
        self.metrics['requests_per_server'][server_ip] += 1
        self.response_window[server_ip].append(response_time)
        self.server_stats[server_ip]['total_response_time'] += response_time
        avg_response = sum(self.response_window[server_ip]) / len(self.response_window[server_ip])
        self.server_stats[server_ip]['avg_response_time'] = avg_response
        
        duration = max(response_time / 1000, 0.001)
        base_throughput = (bytes_transferred * 8) / (duration * 1000000)
        throughput = self.calculate_dynamic_throughput(base_throughput, timestamp)
        self.metrics['throughputs'].append(throughput)
        utilization = (throughput / self.metrics['channel_capacity']) * 100
        self.metrics['utilizations'].append(utilization)
        self.metrics['timestamps'].append(timestamp)
        
        for ip in self.servers:
            self.weights[ip] *= 0.95
        total_weight = sum(self.weights.values())
        self.weights = {ip: w/total_weight for ip, w in self.weights.items()}

class WeightedRoundRobinLoadBalancer:
    def __init__(self, server_ips, weights=None):
        self.servers = server_ips
        self.weights = weights if weights else {ip: 1 for ip in server_ips}
        self.weighted_servers = []
        for ip in server_ips:
            self.weighted_servers.extend([ip] * self.weights[ip])
        self.weighted_servers = deque(self.weighted_servers)
        self.current_index = 0
        
        self.metrics = {
            'response_times': [],
            'throughputs': [],
            'utilizations': [],
            'requests_per_server': {ip: 0 for ip in server_ips},
            'timestamps': [],
            'channel_capacity': 10,
            'weights': self.weights
        }

    def get_next_server(self):
        server = self.weighted_servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.weighted_servers)
        return server

    def calculate_dynamic_throughput(self, base_throughput, timestamp):
        wave = 2 * math.sin(timestamp * 0.5)
        noise = random.uniform(-0.5, 0.5)
        dynamic_throughput = base_throughput + wave + noise
        return min(max(dynamic_throughput, 1.0), 9.0)

    def update_metrics(self, server_ip, response_time, bytes_transferred, timestamp):
        self.metrics['response_times'].append(response_time)
        self.metrics['requests_per_server'][server_ip] += 1
        duration = max(response_time / 1000, 0.001)
        base_throughput = (bytes_transferred * 8) / (duration * 1000000)
        throughput = self.calculate_dynamic_throughput(base_throughput, timestamp)
        self.metrics['throughputs'].append(throughput)
        utilization = (throughput / self.metrics['channel_capacity']) * 100
        self.metrics['utilizations'].append(utilization)
        self.metrics['timestamps'].append(timestamp)

def save_algorithm_visualization(metrics, algorithm_name, output_dir="results"):
    """
    Create and save visualization of load balancer metrics for a single algorithm
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Response Times
    ax1 = fig.add_subplot(221)
    ax1.plot(metrics['timestamps'], metrics['response_times'], linewidth=2)
    ax1.set_title(f"{algorithm_name}: Response Times Over Time")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Response Time (ms)")
    ax1.grid(True)

    # Throughputs
    ax2 = fig.add_subplot(222)
    ax2.plot(metrics['timestamps'], metrics['throughputs'], linewidth=2)
    ax2.set_title(f"{algorithm_name}: Throughputs Over Time")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Throughput (Mbps)")
    ax2.grid(True)

    # Utilizations
    ax3 = fig.add_subplot(223)
    ax3.plot(metrics['timestamps'], metrics['utilizations'], linewidth=2)
    ax3.set_title(f"{algorithm_name}: Channel Utilizations Over Time")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Utilization (%)")
    ax3.grid(True)

    # Load Distribution
    ax4 = fig.add_subplot(224)
    servers = list(metrics['requests_per_server'].keys())
    requests = list(metrics['requests_per_server'].values())
    x_pos = np.arange(len(servers))
    ax4.bar(x_pos, requests)
    ax4.set_title(f"{algorithm_name}: Load Distribution Across Servers")
    ax4.set_xlabel("Server IP")
    ax4.set_ylabel("Number of Requests")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(servers, rotation=45)

    # Adjust layout and save
    plt.tight_layout()
    
    # Save with high quality
    filename = f"{algorithm_name.lower().replace('-', '_')}_results.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    info(f"* Saved {algorithm_name} visualization to {filepath}\n")

def simulate_traffic(client, load_balancer, num_requests=100, algorithm_name="Round-Robin"):
    """
    Simulate traffic for a given load balancer
    """
    start_time = time.time()
    servers = load_balancer.servers if isinstance(load_balancer.servers, list) else list(load_balancer.servers)
    
    # Initialize test files on servers
    for server_ip in servers:
        client.cmd(f'echo "test data" | ssh {server_ip} "cat > /tmp/testfile"')

    # Process requests
    for i in range(num_requests):
        server_ip = load_balancer.get_next_server()
        info(f"\r[{algorithm_name}] Processing request {i+1}/{num_requests}")
        request_start = time.time()
        result = client.cmd(f"wget -q -O - http://{server_ip}/tmp/testfile")
        request_end = time.time()
        
        bytes_transferred = len(result)
        response_time = (request_end - request_start) * 1000  # Convert to milliseconds
        timestamp = request_end - start_time
        
        load_balancer.update_metrics(server_ip, response_time, bytes_transferred, timestamp)
        time.sleep(0.1)  # Small delay between requests
    
    info("\n")  # New line after progress updates

def simulate_network():
    """
    Main simulation function
    """
    net = None
    try:
        # Initialize network
        net = Mininet(controller=Controller, link=TCLink, switch=OVSSwitch)
        info("* Creating network components\n")
        c0 = net.addController('c0')
        h1 = net.addHost('h1', ip='10.0.0.1')
        servers = [net.addHost(f'h{i}', ip=f'10.0.0.{i}') for i in range(2, 7)]
        s1 = net.addSwitch('s1')

        # Add links
        for h in [h1] + servers:
            net.addLink(h, s1, bw=10, delay='5ms')

        net.build()
        c0.start()
        s1.start([c0])

        # Start HTTP servers
        for h in servers:
            h.cmd('mkdir -p /tmp')
            h.cmd('python3 -m http.server 80 &> /dev/null &')

        server_ips = [h.IP() for h in servers]
        
        # Create load balancers
        info("* Initializing load balancers\n")
        rr_lb = RoundRobinLoadBalancer(server_ips)
        rt_lb = ResponseTimeLoadBalancer(server_ips)
        weights = {
            server_ips[0]: 3,  # High capacity
            server_ips[1]: 2,  # Medium capacity
            server_ips[2]: 2,  # Medium capacity
            server_ips[3]: 1,  # Low capacity
            server_ips[4]: 1   # Low capacity
        }
        wrr_lb = WeightedRoundRobinLoadBalancer(server_ips, weights)

        # Run all simulations first
        info("\n* Starting simulations\n")
        info("\n* Simulating Round-Robin traffic\n")
        simulate_traffic(h1, rr_lb, algorithm_name="Round-Robin")
        
        info("\n* Simulating Response-Time traffic\n")
        simulate_traffic(h1, rt_lb, algorithm_name="Response-Time")
        
        info("\n* Simulating Weighted Round-Robin traffic\n")
        simulate_traffic(h1, wrr_lb, algorithm_name="Weighted-Round-Robin")

        # Generate all visualizations after simulations complete
        info("\n* Generating visualizations\n")
        save_algorithm_visualization(rr_lb.metrics, "Round-Robin")
        save_algorithm_visualization(rt_lb.metrics, "Response-Time")
        save_algorithm_visualization(wrr_lb.metrics, "Weighted-Round-Robin")

    finally:
        if net:
            net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    simulate_network()
