from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import numpy as np
import math
import os
from datetime import datetime

class BaseLoadBalancer:
    def __init__(self, server_ips):
        self.servers = server_ips  # Add servers attribute to base class
        self.metrics = {
            'response_times': [],
            'throughputs': [],
            'server_loads': {ip: [] for ip in server_ips},
            'latencies': [],
            'jitter': [],
            'failed_requests': [],
            'success_rate': [],
            'timestamps': [],
            'server_health': {ip: 1.0 for ip in server_ips},  # 1.0 = healthy, 0.0 = failing
            'request_queue_length': [],
            'total_requests': 0,
            'channel_capacity': 10,  # Mbps
            'moving_averages': {
                'response_time': deque(maxlen=10),
                'throughput': deque(maxlen=10)
            }
        }
        
    def calculate_metrics(self, server_ip, response_time, bytes_transferred, timestamp, success):
        # Basic metrics
        self.metrics['timestamps'].append(timestamp)
        self.metrics['total_requests'] += 1
        
        # Response time and latency
        self.metrics['response_times'].append(response_time)
        self.metrics['moving_averages']['response_time'].append(response_time)
        
        # Throughput calculation with network overhead consideration
        packet_overhead = 40  # TCP/IP header overhead in bytes
        total_bytes = bytes_transferred + (bytes_transferred // 1460 + 1) * packet_overhead
        duration = max(response_time / 1000, 0.001)  # Convert ms to seconds
        throughput = (total_bytes * 8) / (duration * 1000000)  # Mbps
        self.metrics['throughputs'].append(throughput)
        self.metrics['moving_averages']['throughput'].append(throughput)
        
        # Server load tracking
        current_load = len(self.metrics['server_loads'][server_ip])
        self.metrics['server_loads'][server_ip].append(current_load + 1)
        
        # Jitter calculation
        if len(self.metrics['response_times']) > 1:
            jitter = abs(response_time - self.metrics['response_times'][-2])
            self.metrics['jitter'].append(jitter)
        else:
            self.metrics['jitter'].append(0)
        
        # Success rate tracking
        self.metrics['failed_requests'].append(0 if success else 1)
        success_rate = 1 - (sum(self.metrics['failed_requests'][-100:]) / 
                           min(len(self.metrics['failed_requests']), 100))
        self.metrics['success_rate'].append(success_rate)
        
        # Server health update
        self.update_server_health(server_ip, response_time, success)
        
        # Queue length estimation - Fixed to handle empty lists
        queue_length = sum(load[-1] if load else 0 for load in self.metrics['server_loads'].values())
        self.metrics['request_queue_length'].append(queue_length)
        
    def update_server_health(self, server_ip, response_time, success):
        health_decay = 0.95 if success else 0.7
        response_penalty = min(response_time / 1000, 0.3)  # Cap penalty at 30%
        
        self.metrics['server_health'][server_ip] *= health_decay
        self.metrics['server_health'][server_ip] -= response_penalty
        self.metrics['server_health'][server_ip] = max(0.1, 
                                                     min(1.0, self.metrics['server_health'][server_ip]))

class RoundRobinLoadBalancer(BaseLoadBalancer):
    def __init__(self, server_ips):
        super().__init__(server_ips)
        self.server_queue = deque(server_ips)

    def get_next_server(self):
        self.server_queue.rotate(-1)
        return self.server_queue[0]

class ResponseTimeLoadBalancer(BaseLoadBalancer):
    def __init__(self, server_ips):
        super().__init__(server_ips)
        self.response_window = {ip: deque(maxlen=10) for ip in server_ips}
        self.weights = {ip: 1.0 for ip in server_ips}

    def get_next_server(self):
        server_scores = {}
        for ip in self.servers:  # Use self.servers instead of self.response_window
            avg_response = (sum(self.response_window[ip]) / len(self.response_window[ip]) 
                          if self.response_window[ip] else 100)
            health_factor = self.metrics['server_health'][ip]
            load_factor = len(self.metrics['server_loads'][ip]) / max(1, self.metrics['total_requests'])
            
            server_scores[ip] = (
                0.4 * avg_response +
                0.3 * (1 - health_factor) * 100 +
                0.3 * load_factor * 100
            ) * self.weights[ip]
        
        selected_server = min(server_scores.items(), key=lambda x: x[1])[0]
        self.adjust_weights(selected_server)
        return selected_server

    def adjust_weights(self, selected_server):
        self.weights[selected_server] *= 0.95
        total_weight = sum(self.weights.values())
        self.weights = {ip: w/total_weight for ip, w in self.weights.items()}

class WeightedRoundRobinLoadBalancer(BaseLoadBalancer):
    def __init__(self, server_ips, weights=None):
        super().__init__(server_ips)  # Call parent constructor first
        self.weights = weights if weights else {ip: 1 for ip in server_ips}
        self.weighted_servers = []
        self.current_index = 0
        self.create_weighted_server_list()

    def create_weighted_server_list(self):
        self.weighted_servers = []
        for ip, weight in self.weights.items():
            health_adjusted_weight = max(1, int(weight * self.metrics['server_health'][ip]))
            self.weighted_servers.extend([ip] * health_adjusted_weight)
        if not self.weighted_servers:  # Failsafe: if all weights are 0, use base weights
            self.weighted_servers = list(self.weights.keys())

    def get_next_server(self):
        if not self.weighted_servers:
            self.create_weighted_server_list()
        
        if not self.weighted_servers:  # Additional safety check
            return self.servers[0]  # Return first server as fallback
            
        server = self.weighted_servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.weighted_servers)
        return server
        
def save_enhanced_visualizations(metrics, algorithm_name, output_dir="results"):
    """Enhanced visualization function with fixed data alignment"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main performance dashboard
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Response Times with Moving Average
    ax1 = fig.add_subplot(331)
    ax1.plot(metrics['timestamps'], metrics['response_times'], 'b-', alpha=0.5, label='Raw')
    if metrics['moving_averages']['response_time']:
        ma_response = list(metrics['moving_averages']['response_time'])
        ax1.plot(metrics['timestamps'][-len(ma_response):], ma_response, 'r-', 
                label='Moving Avg')
    ax1.set_title('Response Times')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Response Time (ms)')
    ax1.legend()
    ax1.grid(True)

    # 2. Throughput Analysis
    ax2 = fig.add_subplot(332)
    ax2.plot(metrics['timestamps'], metrics['throughputs'], 'g-')
    ax2.set_title('Network Throughput')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Throughput (Mbps)')
    ax2.grid(True)

    # 3. Server Health Heatmap
    ax3 = fig.add_subplot(333)
    health_data = [[v for v in metrics['server_health'].values()]]
    sns.heatmap(health_data, ax=ax3, cmap='RdYlGn', 
                xticklabels=list(metrics['server_health'].keys()),
                yticklabels=['Health'])
    ax3.set_title('Server Health Status')

    # 4. Jitter Analysis - Fixed alignment
    ax4 = fig.add_subplot(334)
    if len(metrics['jitter']) > 0:  # Only plot if we have jitter data
        # Ensure timestamps and jitter arrays have the same length
        plot_length = min(len(metrics['timestamps']), len(metrics['jitter']))
        ax4.plot(metrics['timestamps'][:plot_length], metrics['jitter'][:plot_length], 'b-')
    ax4.set_title('Network Jitter')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Jitter (ms)')
    ax4.grid(True)

    # 5. Success Rate
    ax5 = fig.add_subplot(335)
    ax5.plot(metrics['timestamps'], metrics['success_rate'], 'g-')
    ax5.set_title('Request Success Rate')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Success Rate')
    ax5.set_ylim([0, 1.1])
    ax5.grid(True)

    # 6. Queue Length
    ax6 = fig.add_subplot(336)
    ax6.plot(metrics['timestamps'], metrics['request_queue_length'], 'r-')
    ax6.set_title('Request Queue Length')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Queue Length')
    ax6.grid(True)

    # 7. Server Load Distribution
    ax7 = fig.add_subplot(337)
    server_loads = [max(loads) if loads else 0 for loads in metrics['server_loads'].values()]
    ax7.bar(metrics['server_loads'].keys(), server_loads)
    ax7.set_title('Peak Server Loads')
    ax7.set_xlabel('Server')
    ax7.set_ylabel('Peak Load')
    plt.xticks(rotation=45)

    # 8. Response Time Distribution
    ax8 = fig.add_subplot(338)
    sns.histplot(metrics['response_times'], ax=ax8, bins=30, kde=True)
    ax8.set_title('Response Time Distribution')
    ax8.set_xlabel('Response Time (ms)')

    # 9. System Performance Score
    ax9 = fig.add_subplot(339)
    perf_score = (np.mean(metrics['success_rate']) * 0.4 +
                 (1 - np.mean(metrics['response_times']) / max(metrics['response_times'])) * 0.3 +
                 (np.mean(metrics['throughputs']) / metrics['channel_capacity']) * 0.3)
    ax9.text(0.5, 0.5, f'Performance Score:\n{perf_score:.2f}', 
             horizontalalignment='center', verticalalignment='center',
             fontsize=15)
    ax9.axis('off')

    plt.tight_layout()
    filename = f"{algorithm_name.lower().replace('-', '_')}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics summary
    summary_file = os.path.join(output_dir, f"{algorithm_name.lower()}_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Performance Summary for {algorithm_name}\n")
        f.write(f"Average Response Time: {np.mean(metrics['response_times']):.2f} ms\n")
        f.write(f"Average Throughput: {np.mean(metrics['throughputs']):.2f} Mbps\n")
        f.write(f"Average Success Rate: {np.mean(metrics['success_rate']):.2%}\n")
        f.write(f"Average Jitter: {np.mean(metrics['jitter']):.2f} ms\n")
        f.write(f"Overall Performance Score: {perf_score:.2f}\n")

    info(f"* Saved {algorithm_name} visualization to {filepath}\n")
    info(f"* Saved {algorithm_name} summary to {summary_file}\n")

def simulate_traffic(client, load_balancer, num_requests=100, algorithm_name="Round-Robin"):
    """Enhanced traffic simulation with failure scenarios and varying load"""
    start_time = time.time()
    servers = (load_balancer.servers if isinstance(load_balancer.servers, list) 
              else list(load_balancer.servers))
    
    # Initialize test files
    for server_ip in servers:
        client.cmd(f'echo "test data" | ssh {server_ip} "cat > /tmp/testfile"')

    for i in range(num_requests):
        server_ip = load_balancer.get_next_server()
        info(f"\r[{algorithm_name}] Processing request {i+1}/{num_requests}")
        
        # Simulate varying load conditions
        if i % 20 == 0:  # Every 20 requests, simulate high load
            time.sleep(random.uniform(0.2, 0.5))
        
        request_start = time.time()
        result = client.cmd(f"wget -q -O - http://{server_ip}/tmp/testfile")
        request_end = time.time()
        
        # Calculate metrics
        success = len(result) > 0
        response_time = (request_end - request_start) * 1000
        bytes_transferred = len(result) if success else 0
        timestamp = request_end - start_time
        
        # Simulate occasional network issues
        if random.random() < 0.05:  # 5% chance of network issues
            response_time *= 2
            success = False
        
        load_balancer.calculate_metrics(server_ip, response_time, bytes_transferred, 
                                      timestamp, success)
        
        time.sleep(0.1)
    
    info("\n")

def simulate_network():
    """Main simulation function with enhanced network topology"""
    net = None
    try:
        net = Mininet(controller=Controller, link=TCLink, switch=OVSSwitch)
        
        # Create network components
        info("* Creating enhanced network topology\n")
        c0 = net.addController('c0')
        client = net.addHost('h1', ip='10.0.0.1')
        servers = [net.addHost(f'h{i}', ip=f'10.0.0.{i}') for i in range(2, 7)]
        switch = net.addSwitch('s1')

        # Add links with varying characteristics
        net.addLink(client, switch, bw=10, delay='5ms')
        for i, server in enumerate(servers):
            # Simulate different network conditions for each server
            delay = f"{5 + i * 2}ms"
            bw = 10 - i  # Decreasing bandwidth for each server
            net.addLink(server, switch, bw=bw, delay=delay)

        net.build()
        c0.start()
        switch.start([c0])

        # Start HTTP servers with different configurations
        info("* Starting HTTP servers\n")
        for server in servers:
            server.cmd('mkdir -p /tmp')
            # Use different buffer sizes for each server to simulate varying performance
            buffer_size = 1024 * (servers.index(server) + 1)
            server.cmd(f'python3 -m http.server 80 --bind {server.IP()} '
                      f'--buffer-size {buffer_size} &> /dev/null &')

        server_ips = [server.IP() for server in servers]
        
        # Initialize load balancers with different configurations
        info("* Initializing load balancers with different strategies\n")
        
        # Round Robin
        rr_lb = RoundRobinLoadBalancer(server_ips)
        
        # Response Time with initial server weights
        rt_lb = ResponseTimeLoadBalancer(server_ips)
        
        # Weighted Round Robin with capacity-based weights
        wrr_weights = {
            server_ips[0]: 5,  # Highest capacity
            server_ips[1]: 4,
            server_ips[2]: 3,
            server_ips[3]: 2,
            server_ips[4]: 1   # Lowest capacity
        }
        wrr_lb = WeightedRoundRobinLoadBalancer(server_ips, wrr_weights)

        # Run simulations with different load patterns
        info("\n* Starting load balancer simulations\n")
        
        # Simulate normal traffic
        info("\n* Testing Round-Robin under normal load\n")
        simulate_traffic(client, rr_lb, num_requests=100, algorithm_name="Round-Robin")
        
        # Simulate heavy traffic
        info("\n* Testing Response-Time under heavy load\n")
        simulate_traffic(client, rt_lb, num_requests=150, algorithm_name="Response-Time")
        
        # Simulate burst traffic
        info("\n* Testing Weighted-Round-Robin under burst load\n")
        simulate_traffic(client, wrr_lb, num_requests=120, algorithm_name="Weighted-Round-Robin")

        # Generate comprehensive performance reports
        info("\n* Generating performance visualizations and reports\n")
        save_enhanced_visualizations(rr_lb.metrics, "Round-Robin")
        save_enhanced_visualizations(rt_lb.metrics, "Response-Time")
        save_enhanced_visualizations(wrr_lb.metrics, "Weighted-Round-Robin")

        # Compare algorithms
        compare_algorithms(rr_lb, rt_lb, wrr_lb)

    except Exception as e:
        info(f"* Simulation failed: {str(e)}\n")
        raise
    finally:
        if net:
            info("* Cleaning up network\n")
            # Clean up server processes
            for server in servers:
                server.cmd('pkill -f "python3 -m http.server"')
            net.stop()

def compare_algorithms(rr_lb, rt_lb, wrr_lb):
    """Compare the performance of different load balancing algorithms"""
    plt.figure(figsize=(15, 10))
    
    # Response Time Comparison
    plt.subplot(2, 2, 1)
    plt.plot(rr_lb.metrics['timestamps'], rr_lb.metrics['response_times'], 
             label='Round-Robin', alpha=0.7)
    plt.plot(rt_lb.metrics['timestamps'], rt_lb.metrics['response_times'], 
             label='Response-Time', alpha=0.7)
    plt.plot(wrr_lb.metrics['timestamps'], wrr_lb.metrics['response_times'], 
             label='Weighted-RR', alpha=0.7)
    plt.title('Response Time Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Response Time (ms)')
    plt.legend()
    plt.grid(True)

    # Throughput Comparison
    plt.subplot(2, 2, 2)
    plt.plot(rr_lb.metrics['timestamps'], rr_lb.metrics['throughputs'], 
             label='Round-Robin', alpha=0.7)
    plt.plot(rt_lb.metrics['timestamps'], rt_lb.metrics['throughputs'], 
             label='Response-Time', alpha=0.7)
    plt.plot(wrr_lb.metrics['timestamps'], wrr_lb.metrics['throughputs'], 
             label='Weighted-RR', alpha=0.7)
    plt.title('Throughput Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (Mbps)')
    plt.legend()
    plt.grid(True)

    # Success Rate Comparison
    plt.subplot(2, 2, 3)
    plt.plot(rr_lb.metrics['timestamps'], rr_lb.metrics['success_rate'], 
             label='Round-Robin', alpha=0.7)
    plt.plot(rt_lb.metrics['timestamps'], rt_lb.metrics['success_rate'], 
             label='Response-Time', alpha=0.7)
    plt.plot(wrr_lb.metrics['timestamps'], wrr_lb.metrics['success_rate'], 
             label='Weighted-RR', alpha=0.7)
    plt.title('Success Rate Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True)

    # Overall Performance Score
    plt.subplot(2, 2, 4)
    algorithms = ['Round-Robin', 'Response-Time', 'Weighted-RR']
    scores = []
    for lb in [rr_lb, rt_lb, wrr_lb]:
        score = (np.mean(lb.metrics['success_rate']) * 0.4 +
                (1 - np.mean(lb.metrics['response_times']) / 
                 max(lb.metrics['response_times'])) * 0.3 +
                (np.mean(lb.metrics['throughputs']) / 
                 lb.metrics['channel_capacity']) * 0.3)
        scores.append(score)
    
    plt.bar(algorithms, scores)
    plt.title('Overall Performance Score')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig('results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save comparison summary
    with open('results/comparison_summary.txt', 'w') as f:
        f.write("Load Balancer Algorithm Comparison Summary\n")
        f.write("=========================================\n\n")
        
        for algorithm, lb in zip(algorithms, [rr_lb, rt_lb, wrr_lb]):
            f.write(f"\n{algorithm} Statistics:\n")
            f.write(f"Average Response Time: {np.mean(lb.metrics['response_times']):.2f} ms\n")
            f.write(f"Average Throughput: {np.mean(lb.metrics['throughputs']):.2f} Mbps\n")
            f.write(f"Average Success Rate: {np.mean(lb.metrics['success_rate']):.2%}\n")
            f.write(f"Average Jitter: {np.mean(lb.metrics['jitter']):.2f} ms\n")
            f.write(f"Performance Score: {scores[algorithms.index(algorithm)]:.2f}\n")
            f.write("-" * 50 + "\n")

if __name__ == '__main__':
    setLogLevel('info')
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    simulate_network()
