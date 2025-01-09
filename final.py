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
import os
from datetime import datetime

class BaseLoadBalancer:
    def __init__(self, server_ips):
        self.servers = server_ips
        self.metrics = {
            'response_times': [],
            'throughputs': [],
            'server_loads': {ip: [] for ip in server_ips},
            'latencies': [],
            'jitter': [],
            'timestamps': [],
            'server_health': {ip: 1.0 for ip in server_ips},
            'request_queue_length': [],
            'total_requests': 0,
            'channel_capacity': 10,  # Mbps
            'moving_averages': {
                'response_time': deque(maxlen=10),
                'throughput': deque(maxlen=10)
            },
            'success_rate': [],  # Added success rate metric
            'srtc_scores': {ip: [] for ip in server_ips},
            'ldi_history': [],
            'nes_scores': [],
            'rut_metrics': {ip: [] for ip in server_ips},
        }
        
    def calculate_metrics(self, server_ip, response_time, bytes_transferred, timestamp, success=True):
        # Basic metrics
        self.metrics['timestamps'].append(timestamp)
        self.metrics['total_requests'] += 1
        
        # Success rate tracking
        self.metrics['success_rate'].append(1.0 if success else 0.0)
        
        # Response time and latency
        self.metrics['response_times'].append(response_time)
        self.metrics['moving_averages']['response_time'].append(response_time)
        
        # Throughput calculation with network overhead
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
            self.metrics['latencies'].append(jitter)
        else:
            self.metrics['latencies'].append(0)
            
        # Calculate advanced metrics
        self._calculate_srtc(server_ip, response_time)
        self._calculate_ldi()
        self._calculate_nes(throughput, response_time)
        self._calculate_rut(server_ip, response_time, throughput)
        
        # Queue length estimation
        queue_length = sum(load[-1] if load else 0 for load in self.metrics['server_loads'].values())
        self.metrics['request_queue_length'].append(queue_length)

    def _calculate_srtc(self, server_ip, response_time):
        """Calculate Server Response Time Consistency"""
        if self.metrics['server_loads'][server_ip]:
            prev_times = self.metrics['response_times'][-10:]
            std_dev = np.std(prev_times) if len(prev_times) > 1 else 0
            mean_time = np.mean(prev_times)
            consistency_score = 1 / (1 + std_dev/mean_time) if mean_time > 0 else 0
            self.metrics['srtc_scores'][server_ip].append(consistency_score)
        else:
            self.metrics['srtc_scores'][server_ip].append(1.0)

    def _calculate_ldi(self):
        """Calculate Load Distribution Index"""
        total_loads = [len(loads) for loads in self.metrics['server_loads'].values()]
        if sum(total_loads) > 0:
            mean_load = np.mean(total_loads)
            mad = np.mean(np.abs(total_loads - mean_load))
            ldi = 1 - (mad / (2 * mean_load)) if mean_load > 0 else 0
            self.metrics['ldi_history'].append(ldi)
        else:
            self.metrics['ldi_history'].append(1.0)

    def _calculate_nes(self, throughput, response_time):
        """Calculate Network Efficiency Score"""
        normalized_throughput = throughput / self.metrics['channel_capacity']
        normalized_response = 1 / (1 + response_time/1000)
        nes = (0.6 * normalized_throughput + 0.4 * normalized_response)
        self.metrics['nes_scores'].append(nes)

    def _calculate_rut(self, server_ip, response_time, throughput):
        """Calculate Resource Utilization Tracking"""
        max_expected_throughput = self.metrics['channel_capacity']
        normalized_throughput = throughput / max_expected_throughput
        normalized_response = 1 / (1 + response_time/1000)
        utilization = (0.7 * normalized_throughput + 0.3 * (1 - normalized_response))
        self.metrics['rut_metrics'][server_ip].append(utilization)

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
        for ip in self.servers:
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
        super().__init__(server_ips)
        self.weights = weights if weights else {ip: 1 for ip in server_ips}
        self.weighted_servers = []
        self.current_index = 0
        self.create_weighted_server_list()

    def create_weighted_server_list(self):
        self.weighted_servers = []
        for ip, weight in self.weights.items():
            health_adjusted_weight = max(1, int(weight * self.metrics['server_health'][ip]))
            self.weighted_servers.extend([ip] * health_adjusted_weight)
        if not self.weighted_servers:
            self.weighted_servers = list(self.weights.keys())

    def get_next_server(self):
        if not self.weighted_servers:
            self.create_weighted_server_list()
        
        if not self.weighted_servers:
            return self.servers[0]
            
        server = self.weighted_servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.weighted_servers)
        return server

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def save_enhanced_visualizations(metrics, algorithm_name, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig = plt.figure(figsize=(20, 15))
    
    # Response Times with Moving Average
    ax1 = fig.add_subplot(331)
    ax1.plot(metrics['timestamps'], metrics['response_times'], 'b-', alpha=0.5, label='Raw')
    if metrics['moving_averages']['response_time']:
        ma_response = list(metrics['moving_averages']['response_time'])
        ax1.plot(metrics['timestamps'][-len(ma_response):], ma_response, 'r-', label='Moving Avg')
    ax1.set_title('Response Times')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Response Time (ms)')
    ax1.legend()
    ax1.grid(True)

    # Server Response Time Consistency
    ax2 = fig.add_subplot(332)
    for server_ip, scores in metrics['srtc_scores'].items():
        if scores:
            ax2.plot(metrics['timestamps'][-len(scores):], scores, label=f'Server {server_ip}', alpha=0.7)
    ax2.set_title('Server Response Time Consistency')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Consistency Score')
    ax2.legend()
    ax2.grid(True)

    # Load Distribution Index
    ax3 = fig.add_subplot(333)
    if metrics['ldi_history']:
        ax3.plot(metrics['timestamps'][-len(metrics['ldi_history']):], metrics['ldi_history'], 'g-')
    ax3.set_title('Load Distribution Index')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('LDI Score')
    ax3.grid(True)

    # Network Efficiency Score
    ax4 = fig.add_subplot(334)
    if metrics['nes_scores']:
        ax4.plot(metrics['timestamps'][-len(metrics['nes_scores']):], metrics['nes_scores'], 'b-')
    ax4.set_title('Network Efficiency Score')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('NES')
    ax4.grid(True)

    # Fixed Resource Utilization Heatmap
    ax5 = fig.add_subplot(335)
    # Convert the data to a proper numpy array with consistent shapes
    max_len = max(len(util) for util in metrics['rut_metrics'].values())
    rut_data = np.zeros((len(metrics['server_loads']), max_len))
    
    for i, (ip, values) in enumerate(metrics['rut_metrics'].items()):
        rut_data[i, :len(values)] = values
        
    if rut_data.size > 0:
        # Create proper tick labels
        x_ticks = np.linspace(0, max_len-1, min(10, max_len))  # Limit to 10 ticks
        x_labels = [f"{int(x)}" for x in x_ticks]
        y_labels = list(metrics['server_loads'].keys())
        
        sns.heatmap(rut_data, ax=ax5, cmap='YlOrRd',
                    xticklabels=x_labels,
                    yticklabels=y_labels,
                    cbar_kws={'label': 'Resource Utilization'})
        
    ax5.set_title('Resource Utilization Over Time')
    ax5.set_xlabel('Time Index')
    ax5.set_ylabel('Server')

    # Throughput Analysis
    ax6 = fig.add_subplot(336)
    ax6.plot(metrics['timestamps'], metrics['throughputs'], 'g-')
    ax6.set_title('Network Throughput')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Throughput (Mbps)')
    ax6.grid(True)

    # Server Load Distribution
    ax7 = fig.add_subplot(337)
    server_loads = [max(loads) if loads else 0 for loads in metrics['server_loads'].values()]
    ax7.bar(metrics['server_loads'].keys(), server_loads)
    ax7.set_title('Peak Server Loads')
    ax7.set_xlabel('Server')
    ax7.set_ylabel('Peak Load')
    plt.xticks(rotation=45)

    # Response Time Distribution
    ax8 = fig.add_subplot(338)
    sns.histplot(metrics['response_times'], ax=ax8, bins=30, kde=True)
    ax8.set_title('Response Time Distribution')
    ax8.set_xlabel('Response Time (ms)')

    # Overall Performance Score
    ax9 = fig.add_subplot(339)
    avg_nes = np.mean(metrics['nes_scores']) if metrics['nes_scores'] else 0
    avg_ldi = np.mean(metrics['ldi_history']) if metrics['ldi_history'] else 0
    avg_srtc = np.mean([np.mean(scores) if scores else 0 for scores in metrics['srtc_scores'].values()])
    
    performance_metrics = {
        'NES': avg_nes,
        'LDI': avg_ldi,
        'SRTC': avg_srtc
    }
    
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    ax9.pie(performance_metrics.values(), labels=performance_metrics.keys(),
            autopct='%1.1f%%', colors=colors)
    ax9.set_title('Performance Metric Distribution')

    plt.tight_layout()
    filename = f"{algorithm_name.lower().replace('-', '_')}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics summary
    summary_file = os.path.join(output_dir, f"{algorithm_name.lower()}_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Performance Summary for {algorithm_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. Response Time Metrics\n")
        f.write(f"   Average Response Time: {np.mean(metrics['response_times']):.2f} ms\n")
        f.write(f"   Response Time Std Dev: {np.std(metrics['response_times']):.2f} ms\n\n")
        
        f.write("2. Server Response Time Consistency\n")
        for server_ip, scores in metrics['srtc_scores'].items():
            if scores:
                f.write(f"   Server {server_ip}: {np.mean(scores):.3f}\n")
        f.write("\n")
        
        f.write("3. Load Distribution Analysis\n")
        if metrics['ldi_history']:
            f.write(f"   Average LDI Score: {np.mean(metrics['ldi_history']):.3f}\n")
            f.write(f"   LDI Stability (std dev): {np.std(metrics['ldi_history']):.3f}\n")
        f.write("\n")
        
        f.write("4. Network Efficiency\n")
        f.write(f"   Average NES: {np.mean(metrics['nes_scores']) if metrics['nes_scores'] else 0:.3f}\n")
        f.write(f"   Peak Throughput: {max(metrics['throughputs']):.2f} Mbps\n")
        f.write(f"   Average Throughput: {np.mean(metrics['throughputs']):.2f} Mbps\n\n")
        
        f.write("5. Resource Utilization Summary\n")
        for server_ip, utilization in metrics['rut_metrics'].items():
            if utilization:
                f.write(f"   Server {server_ip} Average Utilization: {np.mean(utilization):.2%}\n")

    print(f"* Saved {algorithm_name} visualization to {filepath}")
    print(f"* Saved {algorithm_name} summary to {summary_file}")


def simulate_traffic(client, load_balancer, num_requests=100, algorithm_name="Round-Robin"):
    """Simulate network traffic with enhanced monitoring"""
    start_time = time.time()
    
    # Initialize test files
    for server_ip in load_balancer.servers:
        client.cmd(f'echo "test data" > /tmp/testfile')
        client.cmd(f'scp /tmp/testfile {server_ip}:/tmp/testfile')

    for i in range(num_requests):
        server_ip = load_balancer.get_next_server()
        info(f"\r[{algorithm_name}] Processing request {i+1}/{num_requests}")
        
        # Simulate varying load conditions
        if i % 20 == 0:  # Every 20 requests, simulate high load
            time.sleep(random.uniform(0.2, 0.5))
        
        request_start = time.time()
        result = client.cmd(f'wget -q -O - http://{server_ip}/tmp/testfile')
        request_end = time.time()
        
        # Calculate metrics
        success = len(result) > 0
        response_time = (request_end - request_start) * 1000  # Convert to ms
        bytes_transferred = len(result) if success else 0
        timestamp = request_end - start_time
        
        # Simulate occasional network issues
        if random.random() < 0.05:  # 5% chance of network issues
            response_time *= 2
            success = False
        
        load_balancer.calculate_metrics(server_ip, response_time, bytes_transferred, timestamp, success)
        
        time.sleep(0.1)  # Small delay between requests
    
    info("\n")

def compare_algorithms(rr_lb, rt_lb, wrr_lb):
    """Compare the performance of different load balancing algorithms"""
    plt.figure(figsize=(15, 10))
    
    # Response Time Comparison
    plt.subplot(2, 2, 1)
    plt.plot(rr_lb.metrics['timestamps'], rr_lb.metrics['response_times'], label='Round-Robin', alpha=0.7)
    plt.plot(rt_lb.metrics['timestamps'], rt_lb.metrics['response_times'], label='Response-Time', alpha=0.7)
    plt.plot(wrr_lb.metrics['timestamps'], wrr_lb.metrics['response_times'], label='Weighted-RR', alpha=0.7)
    plt.title('Response Time Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Response Time (ms)')
    plt.legend()
    plt.grid(True)

    # Throughput Comparison
    plt.subplot(2, 2, 2)
    plt.plot(rr_lb.metrics['timestamps'], rr_lb.metrics['throughputs'], label='Round-Robin', alpha=0.7)
    plt.plot(rt_lb.metrics['timestamps'], rt_lb.metrics['throughputs'], label='Response-Time', alpha=0.7)
    plt.plot(wrr_lb.metrics['timestamps'], wrr_lb.metrics['throughputs'], label='Weighted-RR', alpha=0.7)
    plt.title('Throughput Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (Mbps)')
    plt.legend()
    plt.grid(True)

    # Success Rate Comparison
    plt.subplot(2, 2, 3)
    plt.plot(rr_lb.metrics['timestamps'], rr_lb.metrics['success_rate'], label='Round-Robin', alpha=0.7)
    plt.plot(rt_lb.metrics['timestamps'], rt_lb.metrics['success_rate'], label='Response-Time', alpha=0.7)
    plt.plot(wrr_lb.metrics['timestamps'], wrr_lb.metrics['success_rate'], label='Weighted-RR', alpha=0.7)
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
        score = (
            np.mean(lb.metrics['success_rate']) * 0.4 +
            (1 - np.mean(lb.metrics['response_times']) / max(lb.metrics['response_times'])) * 0.3 +
            (np.mean(lb.metrics['throughputs']) / lb.metrics['channel_capacity']) * 0.3
        )
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
            f.write(f"Performance Score: {scores[algorithms.index(algorithm)]:.2f}\n")
            f.write("-" * 50 + "\n")

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
            delay = f"{5 + i * 2}ms"
            bw = 10 - i  # Decreasing bandwidth for each server
            net.addLink(server, switch, bw=bw, delay=delay)

        net.build()
        c0.start()
        switch.start([c0])

        # Start HTTP servers
        info("* Starting HTTP servers\n")
        for server in servers:
            server.cmd('python3 -m http.server 80 &> /dev/null &')

        server_ips = [server.IP() for server in servers]
        
        # Initialize load balancers
        info("* Initializing load balancers\n")
        rr_lb = RoundRobinLoadBalancer(server_ips)
        rt_lb = ResponseTimeLoadBalancer(server_ips)
        wrr_lb = WeightedRoundRobinLoadBalancer(server_ips, {
            server_ips[0]: 5,
            server_ips[1]: 4,
            server_ips[2]: 3,
            server_ips[3]: 2,
            server_ips[4]: 1
        })

        # Run simulations
        info("\n* Starting load balancer simulations\n")
        
        info("\n* Testing Round-Robin under normal load\n")
        simulate_traffic(client, rr_lb, num_requests=100, algorithm_name="Round-Robin")
        
        info("\n* Testing Response-Time under heavy load\n")
        simulate_traffic(client, rt_lb, num_requests=150, algorithm_name="Response-Time")
        
        info("\n* Testing Weighted-Round-Robin under burst load\n")
        simulate_traffic(client, wrr_lb, num_requests=120, algorithm_name="Weighted-Round-Robin")

        # Generate reports
        info("\n* Generating performance visualizations and reports\n")
        save_enhanced_visualizations(rr_lb.metrics, "Round-Robin")
        save_enhanced_visualizations(rt_lb.metrics, "Response-Time")
        save_enhanced_visualizations(wrr_lb.metrics, "Weighted-Round-Robin")
        compare_algorithms(rr_lb, rt_lb, wrr_lb)

    except Exception as e:
        info(f"* Simulation failed: {str(e)}\n")
        raise
    finally:
        if net:
            info("* Cleaning up network\n")
            for server in servers:
                server.cmd('pkill -f "python3 -m http.server"')
            net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    os.makedirs('results', exist_ok=True)
    simulate_network()
