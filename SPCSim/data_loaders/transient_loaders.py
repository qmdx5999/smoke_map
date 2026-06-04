import torch
import numpy as np

class TransientGenerator:
  """
  生成 SPAD（单光子雪崩二极管）相机接收到的理想瞬态信号（光子数随时间的分布），暂不考虑光线在场景中的多次反射（多路径效应）
  """
  def __init__(self, Nr = 1, Nc = 1, N_tbins = 1000, tmax = 100, FWHM = 1,  device = "cpu"):
    """
    tmax：激光的往返时间周期（决定最大探测距离）,默认值100ns
    FWHM：激光脉冲的半高全宽（决定脉冲宽窄），默认值1ns
    """

    r"""Class to generate true transients without considering multi-path

    Args:
        Nr (int): Number of pixel rows in resized frame
        Nc (int): Number of pixel columns in resized frame
        N_tbins (int, optional): Number of discrete time bins dividing the laser time period. Defaults to 1000.
        tmax (int, optional): Laser time period in nano seconds. Defaults to 100.
        FWHM (int, optional): Laser full wave half maximum to decide the laser pulse width. Defaults to 1.
        device (str, optional): Choice of compute device. Defaults to 'cpu'.
    """

    self.N_tbins = N_tbins
    self.Nr = Nr
    self.Nc = Nc
    self.device = device
    self.dmax = 3*1e8*tmax*1e-9/2  # LiDAR 能探测的最远距离（米），tmax为100ns时dmax为15米
    self.bin_size = tmax*1.0/N_tbins  # 单个时间仓对应的实际物理时间长度（ns）
    self.tmax = tmax  # 激光的时间周期（ns），表示激光一次发射 - 接收的总时间范围
    self.FWHM = FWHM  # 半高全宽，对于激光这类高斯脉冲，FWHM 是脉冲峰值（最大值）下降到一半时，脉冲在时间轴上的总宽度
    
    # 高斯分布的 FWHM 与标准差 σ 满足固定公式：FWHM ≈ 2.355 × σ，故FWHM / （2.355 *bin_size）就是
    # 以时间仓数量为单位的高斯脉冲标准差 σ ，若FWHM=1ns、bin_size=0.1ns，
    # 则smooth_sigma = 1/(2.355×0.1) ≈ 4.246（个时间仓）
    # 物理意义：该高斯脉冲的标准差≈4.246 个时间仓，对应物理时间为4.246×0.1=0.4246ns，满足FWHM=2.355×0.4246≈1ns的关系
    self.smooth_sigma = FWHM/(2.355*self.bin_size)
    
    self.smooth_window = (int(self.smooth_sigma*5)//2)*2 + 1
    
    # torch.linspace(start, end, steps) 是 PyTorch 生成均匀间隔一维张量的函数：
    # start：起始值（包含）；end：终止值（包含）；steps：生成的元素总数；
    # 核心逻辑：在[start, end]区间内生成steps个等间隔的数值
    # 示例：torch.linspace(0, 9, 10) → 生成 [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.] 共 10 个元素，间隔为(9-0)/(10-1)=1
    x = torch.linspace(0,self.N_tbins-1,self.N_tbins)  # 生成离散时间仓的索引轴（如N_tbins=1000时，生成[0,1,2,...,999]）
    
    # 原始x：一维张量（shape: [N_tbins]），仅表示单像素的时间仓索引
    # repeat(Nr, Nc, 1)：维度扩展（批量适配所有像素）：
    # 第 1 维（行）重复Nr次（对应图像行数）；第 2 维（列）重复Nc次（对应图像列数）；第 3 维（时间仓）重复 1 次（保持索引不变）
    # to(self.device)：将张量移到指定设备（CPU/GPU）；
    # 最终 shape：[Nr, Nc, N_tbins]，即每个像素都有一份相同的时间仓索引轴
    self.x = x.repeat(self.Nr,self.Nc,1).to(self.device)
    
    # 生成全 0 张量，shape 为[Nr, Nc, N_tbins]（与self.x维度一致）
    self.t = torch.zeros((self.Nr,self.Nc, self.N_tbins)).to(self.device)


  def get_signal_attenuation(self, albedo, dist):
    """
    计算激光回波信号衰减因子的核心方法，基于目标反照率和距离平方反比定律模拟真实世界的光强衰减，
    并通过维度调整为后续与三维瞬态脉冲的广播运算做准备
    albedo：场景反照率图（每个像素的反射能力，0-1 之间，1 为完全反射）
    dist：场景真实距离图（每个像素到相机的距离，单位：米）
    signal_attn：衰减后的信号因子（用于后续调制激光脉冲强度）
    """
    
    r"""Method to calculate attenuation factor from albedo information and dist information.

    Args:
        albedo (torch.tensor): Scene albedo image of dimension (Nr, Nc)
        dist (torch.tensor): Scene ground truth distance image of dimension (Nr, Nc)

    Returns:
        signal_attn (torch.tensor) : Attenuated signal
    """

    signal_attn = torch.divide(albedo*1.0,dist**2)  # torch.divide(a, b)：逐元素执行 a/b，得到每个像素的衰减因子
    self.temp_alpha = signal_attn.clone()
    
    # 后续生成的瞬态脉冲是三维张量（形状：(Nr, Nc, N_tbins)，即 “行 × 列 × 时间仓”），
    # 而当前的 signal_attn 是二维张量（形状：(Nr, Nc)）
    # 为了让二维的衰减因子能与三维的脉冲逐像素、逐时间分仓地相乘（广播运算），
    # 需要将 signal_attn 扩展为三维张量，在最后一维增加一个长度为 1 的维度
    signal_attn = signal_attn.reshape(signal_attn.shape[0],signal_attn.shape[1],1)
    
    return signal_attn

  def gt_shift_idx(self, gt_dist):
    """
    将场景真实距离映射为时间仓索引
    gt_dist：场景真实距离图（每个像素到相机的直线距离，单位：米）
    shift_idx：每个像素对应的激光脉冲峰值在时间分仓中的索引（离散坐标）
    """
    
    r"""Method to compute the peak location in bins unit for given scene distance

    Args:
        gt_dist (torch.tensor): Scene ground truth distance image of dimension (Nr, Nc)

    Returns:
        shift_idx (torch.tensor) : Tensor consisting of integer values corresponding to the peak location.
    """

    # 距离归一化（gt_dist*1.0 / self.dmax），将每个像素的绝对距离转换为 LiDAR 能探测的最远距离的比例
    # 映射到时间分仓总数（* self.N_tbins），将归一化的距离比例线性映射到离散时间分仓的索引范围
    # 向下取整（torch.floor(...)），将连续的分仓索引离散化为整数
    shift_idx = torch.floor((gt_dist*1.0/self.dmax)*self.N_tbins).to(device = self.device,
                                                                      dtype = torch.int32)

    return shift_idx

  def get_shifted_laser_pulse_mesh(self, gt_dist):
    """
    生成距离相关的三维高斯脉冲网格的核心方法，它根据每个像素的真实距离将标准高斯脉冲平移到对应时间分仓，
    并进行归一化，最终输出每个像素对应一个时间域高斯脉冲的三维张量
    tr：三维高斯脉冲网格（每个像素对应一个归一化的时间域高斯脉冲）(Nr, Nc, N_tbins)
    """
    
    r"""Method to compute the time shifted gaussian pulse based on the true distance

    Args:
        gt_dist (torch.tensor): Scene ground truth distance image of dimension (Nr, Nc)

    Returns:
        tr (torch.tensor): Tensor of time shifted laser pulse for each pixel based on the true distance
    """
    mu = (self.gt_shift_idx(gt_dist)).long()  # 得到每个像素的脉冲峰值在时间仓中的整数索引，mu是每个像素的高斯脉冲中心位置
    mu = torch.reshape(mu, (self.Nr, self.Nc, 1)).to(device = self.device, dtype = torch.int32)  # 调整维度以适配广播运算
    self.temp_range_bins = mu.clone()
    tr = (torch.exp(-((self.x - mu)**2)/(2*self.smooth_sigma**2)))/(self.smooth_sigma*np.sqrt(2*np.pi))  # 生成未归一化的高斯脉冲
    sum_ = torch.sum(tr, 2)
    sum_ = torch.reshape(sum_, (self.Nr, self.Nc, 1))
    tr = torch.divide(tr, torch.sum(tr, axis=2, keepdims=True))  # 归一化高斯脉冲
    
    return tr


  def get_transient(self, gt_dist, albedo, intensity, alpha_sig, alpha_bkg):
    r"""Method to add noise and attenuation to ideal transient

    Args:
        gt_dist (torch.tensor): Scene ground truth distance image of dimension (Nr, Nc)
        albedo (torch.tensor): Scene albedo image of dimension (Nr, Nc)
        intensity (torch.tensor): Scene intensity image of dimension (Nr, Nc)
        alpha_sig (float): Average signal photons per laser cycle
        alpha_bkg (float): Average background photons per laser cycle (not per bin)

    .. note:: Signal attenuation only depends on scene albedo and background flux depends on 
              the total scene intensity.
    

    Returns:
        r_t1 (torch.tensor): Time shifted attenuated laser pulse representing the photon density incident on the SPAD camera.
    """

    assert (gt_dist.shape[0] == self.Nr) and (gt_dist.shape[0] == self.Nr), \
      "Incorrect initialization of Nr and Nc for TransientGenerator \n must be %d, %d"%(gt_dist.shape[0], gt_dist.shape[1])

    r_t = self.get_shifted_laser_pulse_mesh(gt_dist)  # 生成理想移位高斯脉冲概率分布，形状:(Nr, Nc, N_tbins)，每个像素的时间轴总和为1
    self.signal_attn = self.get_signal_attenuation(albedo/torch.mean(albedo), gt_dist)  # 计算信号衰减因子
    self.bkg_attn = (intensity/torch.mean(intensity)).reshape(albedo.shape[0],albedo.shape[1],1)  # 计算背景衰减因子
    self.k_signal = (self.signal_attn*alpha_sig/torch.mean(self.signal_attn)).to(self.device)  # 每个像素的总信号光子数
    self.k_bkg = (self.bkg_attn*alpha_bkg).to(self.device)  # 计算每个像素的总背景光子数
    
    # 生成最终瞬态信号（信号 + 背景）
    # 信号部分：torch.multiply(r_t, self.k_signal)，信号光子的概率分布 * 每个像素的总信号光子数得到每个时间仓的期望信号光子数
    # 背景部分：self.k_bkg / self.N_tbins，将总背景光子数均匀分到每个时间分仓
    # 相加：信号 + 背景，得到每个分仓的总期望光子数
    # 所以r_t1是泊松分布的期望值
    r_t1 = torch.multiply(r_t, self.k_signal) + self.k_bkg/self.N_tbins
    
    return r_t1