import os
import numpy as np
from data.synthesis import gen
from tqdm import tqdm
import matplotlib.pyplot as plt



class ArgosDPTimeSeriesGenerator(object):
    def __init__(self, counts=64):
        self.counts = counts
        self.extra_counts = 10
        self.times = self.generate_times()
        self.series_dict = dict()   
        
    def generate_times(self):
        times = list()
        for i in range(60):
            if i < 10:
                times.append(f"0{i}")
            else:
                times.append(f"{i}")
        labels = []
        for i in range(60):
            labels.append(f"08:{times[i]}")
        for i in range(60):
            labels.append(f"09:{times[i]}")
        for i in range(60):
            labels.append(f"10:{times[i]}")
        
        return labels
        
    def get_rise_base(self, k=0.01, b=0.2, counts=128):
        samples = list()
        for i in range(counts):
            samples.append(k * i + b)
        return np.array(samples)
        
    def get_decline_base(self, k=-0.01, b=1.5, counts=128):
        samples = list()
        for i in range(counts):
            samples.append(k * i + b)
        return np.array(samples)
    
    def get_noise(self, k=0.01, counts=128):
        samples = list()
        for _ in range(counts):
            coef = 1 if np.random.randint(0, 2) == 0 else -1
            samples.append(coef * np.random.rand() * k)
        return np.array(samples)
    
    def get_line_base(self, counts=128, value=0.5):
        samples = list()
        for i in range(counts):
            samples.append(value)
        return np.array(samples)
    
    
    def visualize(self, x1, x2, save_file):
        plt.figure(figsize=(12, 6))
    
        plt.plot(self.times[:len(x1)], x1, label='Current', color='red')
        plt.scatter(self.times[:len(x1)], x1, color='red', s=20, marker='o')
        plt.plot(self.times[:len(x2)], x2, label='Reference', color='blue')
        plt.scatter(self.times[:len(x2)], x2, color='blue', s=20, marker='o')
        plt.grid(True)
        
        plt.legend(loc='upper center', ncol=3,  bbox_to_anchor=(0.5, 1.08), prop={'weight': 'bold'}, frameon=False)
        plt.xlabel('Time(HH:MM)')
        plt.ylabel('Value')
        plt.xticks(rotation=45, ticks=self.times[:len(x2)][::8])
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()
        
        fig_name = save_file.split("/")[-1].split(".")[0]
        self.series_dict[fig_name] = {"current": x1.tolist(), "reference": x2.tolist()}
        
    
    def up(self, counts=512, save_folder="resources/lines"):
        '''
        有交点陡升且未回落。前期与基线重合，随后出现明显的爬坡或阶跃上涨，并持续维持在高位。
        1. 告警前水位一致，告警后出现显著上涨，且上涨后一直保持高水位
        2. 告警后水位差距比告警前的显著增大
        '''
        sub_counts = counts // 2
        for i in tqdm(range(sub_counts)):
            self.up_consistently_high(i, save_folder)
            self.up_obviously_increase(i, save_folder)
       
       
    def up_consistently_high(self, index, save_folder="resources/lines"):
        '''
        1. 告警前水位一致，告警后出现显著上涨，且上涨后一直保持高水位
        '''
        pos = np.random.randint(8, 16)
        noise_k = np.random.rand() * 0.1
        x1 = self.get_line_base(counts=64)
        y1 = self.get_noise(counts=len(x1), k=noise_k)  # k = 0.05
        x1[-pos:] += 0.5
        z1 = x1 + y1
        
        x2 = self.get_line_base(counts=64)
        y2 = self.get_noise(counts=len(x2), k=noise_k)
        z2 = x2 + y2
    
        self.visualize(z1, z2, f"{save_folder}/up_consistently_high_{index}.png")
    
    
    def up_obviously_increase(self, index, save_folder="resources/lines"):
        '''
        2. 告警后水位差距比告警前的显著增大
        '''
        pos = np.random.randint(16, 32)
        noise_k = np.random.rand() * 0.1
        x1 = self.get_line_base(counts=64)
        y1 = self.get_noise(counts=len(x1), k=noise_k)  # k = 0.05
        
        delta = self.get_noise(counts=pos, k=np.random.rand() * 0.25)
        x1[-pos:] += (np.linspace(0.1, 0.5, pos) + delta)
        z1 = x1 + y1
        
        x2 = self.get_line_base(counts=64)
        y2 = self.get_noise(counts=len(x2), k=noise_k)
        z2 = x2 + y2
    
        self.visualize(z1, z2, f"{save_folder}/up_obviously_increase_{index}.png")
        
    
    def down(self, counts=516, save_folder="resources/lines"):
        '''
        有交点陡降且未回升。前期与基线重合，随后出现明显的下滑或断崖式下跌，并持续维持在低位。
        1. 告警前水位与历史同期一致，告警后出现显著下跌，且下跌后一直保持低水位
        2. 当前曲线下降明显，参考曲线波动不大
        3. 当前曲线与参考曲线都上涨或者下跌，当前曲线在参考曲线下方，且差距值逐渐增大
        '''
        sub_counts = counts // 3
        for i in tqdm(range(sub_counts)):
            self.down_consistently_low(i, save_folder)
            self.down_obviously_decrease(i, save_folder)
            self.down_gap_increase(i, save_folder)
        
    def down_consistently_low(self, index, save_folder="resources/lines"):
        '''
        1. 告警前水位与历史同期一致，告警后出现显著下跌，且下跌后一直保持低水位

        '''
        types = ['normal', 'up', 'down']
        line_type = np.random.choice(types)
        
        if line_type == 'normal':
            x1 = self.get_line_base(counts=64, value=0.8)
            x2 = self.get_line_base(counts=64, value=0.8)
        
        elif line_type == 'up':
            x1 = self.get_rise_base(counts=64)
            x2 = self.get_rise_base(counts=64)
        
        else:
            x1 = self.get_decline_base(counts=64)
            x2 = self.get_decline_base(counts=64)
        
        pos = np.random.randint(8, 32)
        noise_k = np.random.rand() * 0.1
        y1 = self.get_noise(counts=len(x1), k=noise_k)  # k = 0.05
        x1[-pos:] -= 0.5
        z1 = x1 + y1
        
        y2 = self.get_noise(counts=len(x2), k=noise_k)
        z2 = x2 + y2
    
        self.visualize(z1, z2, f"{save_folder}/down_consistently_low_{index}.png")
    
    
    def down_obviously_decrease(self, index, save_folder="resources/lines"):
        '''
        2. 当前曲线下降明显，参考曲线波动不大
        '''
        pos = np.random.randint(16, 32)
        noise_k = np.random.rand() * 0.1
        x1 = self.get_line_base(counts=64, value=0.8)
        y1 = self.get_noise(counts=len(x1), k=noise_k)  # k = 0.05
        
        delta = self.get_noise(counts=pos, k=np.random.rand() * 0.25)
        x1[-pos:] -= (np.linspace(0.1, 0.5, pos) + delta)
        z1 = x1 + y1
        
        x2 = self.get_line_base(counts=64, value=0.8)
        y2 = self.get_noise(counts=len(x2), k=noise_k)
        z2 = x2 + y2
    
        self.visualize(z1, z2, f"{save_folder}/down_obviously_decrease_{index}.png")
        
    def down_gap_increase(self, index, save_folder="resources/lines"):
        '''
        3. 当前曲线与参考曲线都上涨或者下跌，当前曲线在参考曲线下方，且差距值逐渐增大，
        '''
        types = ['up', 'down']
        line_type = np.random.choice(types)
         
        if line_type == 'up':
            x1 = self.get_rise_base(counts=64, k=0.01)
            x2 = self.get_rise_base(counts=64, k=0.02)
        else:
            x1 = self.get_decline_base(counts=64, k=-0.02)
            x2 = self.get_decline_base(counts=64, k=-0.01)
        
        noise_k = np.random.rand() * 0.1
        y1 = self.get_noise(counts=len(x1), k=noise_k)  # k = 0.05
        z1 = x1 + y1
        
        y2 = self.get_noise(counts=len(x2), k=noise_k)
        z2 = x2 + y2
    
        self.visualize(z1, z2, f"{save_folder}/down_gap_increase_{index}.png")
    
    
    def spike_gen(self, counts=512, save_folder="resources/lines"):
        for i in tqdm(range(counts)):
            self.spike(i, save_folder)
            
    
    def spike(self, index, save_folder="resources/lines"):
        '''
        短暂偏离即回落。仅出现 1~2 次极短时的剧烈突变（突增/降），随后迅速回落并与基线重合。
        1.不区分上尖刺和下尖刺, 单点尖刺
        '''
        noise_k = np.random.rand() * 0.1
        x1 = self.get_line_base(counts=64)
        y1 = self.get_noise(counts=len(x1), k=noise_k)  # k = 0.05
        
        spike_pos = list()
        for i in range(7):
            start = i * 8
            end = start + 7
            start = 1 if start == 0 else start
            spike_pos.append(np.random.randint(start, end))
        
        counts = np.random.randint(1, 5)
        spike_pos_chosen = np.random.choice(spike_pos, counts, replace=False)
        
        coef = 1 if np.random.randint(0,2) == 0 else -1
        x1[spike_pos_chosen] += coef * 0.5
        z1 = x1 + y1
        
        x2 = self.get_line_base(counts=64)
        counts = np.random.randint(1, 5)
        spike_pos_chosen = np.random.choice(spike_pos, counts, replace=False)
        x2[spike_pos_chosen] += coef * 0.45
        y2 = self.get_noise(counts=len(x2), k=noise_k)
        z2 = x2 + y2
    
        self.visualize(z1, z2, f"{save_folder}/spike_{index}.png")
        
    
    def same_trend(self, counts=512, save_folder="resources/lines"):
        '''
        1. 趋势接近，两条曲线基本重合
        2. 两条曲线都剧烈振荡，交织在一起，看不出明显的上升和下降趋势
        '''
        sub_counts = counts
        for i in tqdm(range(sub_counts)):
            self.same_trend_normal(i, save_folder)
    
    
    def same_trend_normal(self, index, save_folder="resources/lines"):
        '''
        1. 趋势接近，两条曲线基本重合
        2. 两条曲线都剧烈振荡，交织在一起，看不出明显的上升和下降趋势
        '''
        types = ['normal', 'up', 'down']
        line_type = np.random.choice(types)
        
        if line_type == 'normal':
            x1 = self.get_line_base(counts=64, value=0.8)
            x2 = self.get_line_base(counts=64, value=0.8)
        
        elif line_type == 'up':
            x1 = self.get_rise_base(counts=64)
            x2 = self.get_rise_base(counts=64)
        
        else:
            x1 = self.get_decline_base(counts=64)
            x2 = self.get_decline_base(counts=64)
        
        noise_k = np.random.rand() * 0.01
        y1 = self.get_noise(counts=len(x1), k=noise_k)  # k = 0.05
        z1 = x1 + y1
        
        y2 = self.get_noise(counts=len(x2), k=noise_k)
        z2 = x2 + y2 
    
        self.visualize(z1, z2, f"{save_folder}/normal_{index}.png")
    
    
    def offset(self, counts=512, save_folder="resources/lines"):
        '''
        1. 全程平行且无交点（少于三个交点）。走势无剧烈起伏，但自始至终在基线的上方或下方，存在持续且稳定的差值。
        '''
        sub_counts = counts
        for i in tqdm(range(sub_counts)):
            self.offset_parallel(i, save_folder)
    
    def offset_parallel(self, index, save_folder="resources/lines"):
        '''
        1. 全程平行且无交点（少于三个交点）。走势无剧烈起伏，但自始至终在基线的上方或下方，存在持续且稳定的差值。
        '''
        
        types = ['normal', 'up', 'down']
        line_type = np.random.choice(types)
        
        if line_type == 'normal':
            x1 = self.get_line_base(counts=64, value=0.8)
            x2 = self.get_line_base(counts=64, value=0.8)
        
        elif line_type == 'up':
            x1 = self.get_rise_base(counts=64)
            x2 = self.get_rise_base(counts=64)
        
        else:
            x1 = self.get_decline_base(counts=64)
            x2 = self.get_decline_base(counts=64)
        
        noise_k = np.random.rand() * 0.1
        y1 = self.get_noise(counts=len(x1), k=noise_k)  # k = 0.05
        z1 = x1 + y1
        
        offset_len = (np.random.rand() - 0.5) * 0.5
        y2 = self.get_noise(counts=len(x2), k=noise_k)
        z2 = x2 + y2 + offset_len
        
        self.visualize(z1, z2, f"{save_folder}/offset_{index}.png")
    
    
    def fluctuate(self, index, save_folder="resources/lines"):
        '''
        密集锯齿状跳动。发生 3 次及以上的剧烈起伏，持续不稳定，无法与基线保持贴合。
        '''
        
        noise_k1 = np.random.rand() * 0.2
        x1 = self.get_line_base(counts=64, value=0.5)
        y1 = self.get_noise(counts=len(x1), k=noise_k1)  # k = 0.05
        z1 = x1 + y1
        
        deltas = np.array([0.5,0.2,0.4,0.3,0.2])
        pos = np.random.choice(np.arange(2, 52), 1, replace=False)
        z1[pos[0]:pos[0]+len(deltas)] += deltas

        x2 = self.get_line_base(counts=64, value=0.5)
        noise_k2 = np.random.rand() * 0.5
        y2 = self.get_noise(counts=len(x2), k=noise_k2)
        z2 = x2 + y2
    
        self.visualize(z1, z2, f"{save_folder}/fluctuate_{index}.png")
        
        
    def fluctuate_gen(self, counts=512, save_folder="resources/lines"):
        '''
        1. 全程平行且无交点（少于三个交点）。走势无剧烈起伏，但自始至终在基线的上方或下方，存在持续且稳定的差值。
        '''
        sub_counts = counts 
        for i in tqdm(range(sub_counts)):
            self.fluctuate(i, save_folder)
            
    
    def other(self):
        '''
        其它类别
        '''
        pass
    
    def run(self, save_folder):
        self.up(save_folder=save_folder, counts=256)
        self.down(save_folder=save_folder, counts=256)
        self.spike_gen(save_folder=save_folder, counts=256)
        self.same_trend(save_folder=save_folder, counts=256)
        self.fluctuate_gen(save_folder=save_folder, counts=256)
        self.offset(save_folder=save_folder, counts=256)
        
        import json
        with open(os.path.join(save_folder, "series.json"), "w") as f:
            json.dump(self.series_dict, f, indent=4)
        

if __name__ == '__main__':
    generator = ArgosDPTimeSeriesGenerator()
    
    save_folder = "datas/aiops/files/gen"
    generator.run(save_folder)
   
