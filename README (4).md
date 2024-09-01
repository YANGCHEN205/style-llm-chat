
### 模型简介

稷丰•首个开源中文农业多模态大模型是由山西农业大学研发，以[MiniCPM-Llama3-V 2.5](https://github.com/OpenBMB/MiniCPM-V)为底座，能够从图像、文本、气象数据等多源信息中提取有用信息，为农业生产提供全面、精准的智能化解决方案。我们致力于将稷丰应用于作物健康监测、病虫害识别、土壤肥力分析、农田管理优化等多个方面，帮助农民提升生产效率，减少资源浪费，促进农业的可持续发展。

### 重要说明

1. 本项目仍在不断迭代更新，考虑到农业领域覆盖广泛，图像数据的复杂性，我们需要不断优化指令集；
2. 本项目所提供的数据与模型仅供科研使用，严禁用于商业用途；

### 新闻

- [**2024.06.08**] 👋 👋 👋 稷丰-v1.0版本模型发布，目前模型已支持10种作物病害的多模态检测与决策推荐；

### 更新计划

- [ ] 覆盖更多作物品种，涵盖更多应用领域；
- [ ] 开源预训练数据以及指令微调数据;
- [ ] 发布AgriAgent技术评测报告;
- [ ] 集成Ollama+Open WebUI，优化Web端展示美观性；

### 数据集说明

* 支持作物类型包括：苹果/樱桃/玉米/葡萄/柑桔/桃/辣椒/马铃薯/草莓/番茄;
* 支持作物病害类型包括：黑星病/灰斑病/雪松锈病/白粉病/叶斑病/黑腐病/轮斑病/黄龙病/早疫病/晚疫病/叶枯病/斑点病/斑枯病/黄化曲叶病毒病等；
* 文本数据部分，我们基于[GLM-3-Turbo](https://github.com/THUDM/ChatGLM3)进行指令集构建，部分图像数据来自于[这里](https://github.com/xungeer29/AI-Challenger-Plant-Disease-Recognition);
* 我们欢迎与业界人士的交流与合作，如有数据集提供或者合作事宜，请与zhiweihu@whu.edu.cn进行联系；

### 部分测试结果展示

### 模型训练

1. 开发环境搭建
   ```python
   conda create -n agriagent python=3.10
   conda activate agriagent
   cd AgriAgent
   pip install -r requirements.txt
   ```
2. 指令集构建

* 我们采用的指令集格式样例如下：
  ```python
  {
    "id": "21039",
    "image": "f49c4a3a657daf0f0fedcf63b2c816d4.jpg",
    "conversations": [{
    	"role": "user",
    	"content": "<image>\n你知道这是什么植物吗？如果这株植物状态不好，可能是哪些病害造成的？"
    }, {
    	"role": "assistant",
    	"content": "该图片中的病害属于:严重性玉米灰斑病"
    }, {
    	"role": "user",
    	"content": "对于该病害，最可能造成的原因有哪些，请分点罗列出来。"
    }, {
    	"role": "assistant",
    	"content": "对于玉米灰斑病是一种严重的叶部病害，主要由真菌Cercospora zeae-maydis和Cercospora zeina侵染引起。其严重性可能由以下原因造成：\n\n1. 气候条件：玉米灰斑病的发生与气候密切相关。多雨、空气湿度大、光照差、雾日多的天气有利于病原菌的侵染和病害的发展。特别是在7～8月份，气温较高且降雨频繁，更易导致病害的流行。\n\n2. 田间病原菌积累：病原菌主要以子座或菌丝随病残体越冬，成为翌年初侵染源。病原菌在田间传播迅速，尤其是在玉米种植密度较大、荫蔽的环境中，病原菌容易在植株间传播。\n\n3. 玉米品种抗性差异：不同玉米品种对灰斑病的抗性存在差异。感病品种更容易受到灰斑病的侵染，而抗病品种则能较好地抵抗病原菌的攻击。\n\n4. 栽培管理不当：种植密度过大、偏施氮肥、不施底肥和磷钾肥等不当的栽培管理措施，都会加重玉米灰斑病的发生。\n\n5. 缺乏有效的防治方法：虽然目前已有一些防治方法，如选用抗病品种、加强栽培管理、清洁田园等，但仍有待进一步提高和完善。\n\n6. 病原菌的分生孢子重复侵染：病原菌的分生孢子可以在植株间重复侵染，导致病斑不断扩展蔓延，增加了病害的控制难度。\n\n7. 免疫信号传导机制：研究发现，植物模式识别受体（Plant Pattern Recognition Receptors, PRRs）是一类位于细胞质膜上的受体蛋白，能感知病原体入侵并触发寄主免疫响应。然而，对于玉米灰斑病，植物对病原菌的识别和免疫响应机制仍不十分清楚，这使得病害的防治面临一定的挑战。"
    }, {
    	"role": "user",
    	"content": "对于该病害，我们应该采用哪些措施进行处理，请分点罗列出来。"
    }, {
    	"role": "assistant",
    	"content": "针对严重性的玉米灰斑病，可以采取以下措施进行处理：\n\n1. 清除病残体：在收获后及时清除田间的病残体，包括枯叶和残株，以减少病菌在土壤和环境中的存活和传播。\n\n2. 种植抗病品种：选择对玉米灰斑病具有抗性或耐病性的品种进行种植，这可以有效降低病害的发生率和危害程度。\n\n3. 调整作物布局：进行大面积轮作或间作，避免连续种植玉米，减少病原菌在玉米上的积累。\n\n4. 田间管理：加强田间管理，合理控制种植密度，避免过密种植导致植株间通风透光不良；及时进行中耕除草，改善土壤通气性和水分状况。\n\n5. 合理施肥：避免偏施氮肥，应实施平衡施肥，注重施用磷钾肥，以增强植株的抗病能力。\n\n6. 药剂防治：在发病初期，根据病害情况选择合适的化学药剂进行防治，可喷洒如75%百菌清可湿性粉剂500倍液、50%多菌灵可湿性粉剂600倍液等药剂。根据防治效果和成本选择合适的药剂和浓度，并按照推荐的使用方法进行喷施。\n\n7. 监测和预警：建立病害监测体系，定期对玉米田进行病害监测，及时发现病害发生迹象，并采取相应的防治措施。\n\n8. 农业技术培训：加强对农民的农业技术培训，提高他们对玉米灰斑病防治的认识和能力，确保防治措施得到有效实施。\n\n这些措施应结合实际情况灵活运用，可能需要根据当地的具体气候条件、土壤类型、玉米品种和病害发生情况来调整和优化防治策略"
    }]
  }
  ```
* 如需获取AgriAgent微调指令集，请发送邮件至zhiweihu@whu.edu.cn与我们联系，目前我们仍然在不断扩充指令集，最终版会在适时公开；

3. 模型微调

* 进入finetune/finetune_lora.sh，修改下面内容：
  ```python
  MODEL="openbmb/MiniCPM-Llama3-V-2_5" # 修改成你自己的MiniCPM-Llama3-V-2_5预训练权重位置
  DATA="path/to/trainging_data" # 训练集位置，必须是json格式
  EVAL_DATA="path/to/test_data" # 测试集位置，必须是json格式
  LLM_TYPE="llama3"
  ```
* 执行
  ```python
  sh finetune_lora.sh
  ```
* 如果在训练过程中出现显存溢出问题，可以通过调整finetune_lora.sh参数进行解决：
  * --model_max_length：减小模型最大输入长度限制
  * --scale_resolution：降低图像分辨率
  * --tune_vision false：设置为false

4. Web页面测试

* 执行下面命令，并在浏览器中输入执行命令中出现的网址以访问web界面
  ```python
  python web_demo_2.5.py --device cuda
  ```
* 在执行前，你需要修改web_demo_2.5.py里面的部分内容：
  * 第27行，修改model_path为你自己微调后的模型位置，默认模型的存储位置是finetune/output，建议使用绝对地址；
  * 倒数第1行，修改server_name为你自己的IP地址

### 项目参与者

1. 本项目由山西农业大学开发完成，项目主要开发人员：[胡志伟](https://zhiweihu1103.github.io)
2. 若有相关使用需求或者相关数据集提供，欢迎与我们取得联系：zhiweihu@whu.edu.cn

### 致谢

本项目基于现有开源项目二次开发，在此对相关项目和研发人员表示感谢。

* [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)

## Star History

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=zhiweihu1103/AgriAgent&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=zhiweihu1103/AgriAgent&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=zhiweihu1103/AgriAgent&type=Date" />
</picture>
