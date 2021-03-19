# 说明

这个代码是完全重写了dataset，参照着其他代码自己来实现的，很清楚里面的每一个步骤，使得对dataset，dataloader更加深刻

另外，这个代码里面添加了shuffle，所以训练起来更好，更加稳定，之前的代码没有加打乱，训练起来效果非常不好

结构 bert+BiLSTM+加上loss权重

这个代码是在之前的基础之上，使用伪标签和原来的标签进行混合，然后再重新训练的一个模型，结果如下：

第一次测试（中间断了，只跑了3个epoch）：
    
    Epoch [3/5]
    Iter:   6100,  Train Loss: 0.027,  Train Acc: 100.00%,  Val Loss:  0.26,  Val Acc: 92.96%,  Time: 5:50:17 
    Iter:   6200,  Train Loss:  0.14,  Train Acc: 100.00%,  Val Loss:  0.26,  Val Acc: 92.99%,  Time: 5:55:53 
    Iter:   6300,  Train Loss: 0.007,  Train Acc: 100.00%,  Val Loss:  0.26,  Val Acc: 93.26%,  Time: 6:01:35 *
    Iter:   6400,  Train Loss: 0.012,  Train Acc: 100.00%,  Val Loss:  0.25,  Val Acc: 93.29%,  Time: 6:07:17 *
    Iter:   6500,  Train Loss:  0.16,  Train Acc: 100.00%,  Val Loss:  0.26,  Val Acc: 93.30%,  Time: 6:13:00 *
    Iter:   6600,  Train Loss:  0.58,  Train Acc: 87.50%,  Val Loss:  0.28,  Val Acc: 92.92%,  Time: 6:18:36 
    Iter:   6700,  Train Loss: 0.027,  Train Acc: 100.00%,  Val Loss:  0.24,  Val Acc: 93.37%,  Time: 6:24:19 *
    Iter:   6800,  Train Loss: 0.062,  Train Acc: 100.00%,  Val Loss:  0.24,  Val Acc: 93.01%,  Time: 6:30:00 
    Iter:   6900,  Train Loss:  0.23,  Train Acc: 87.50%,  Val Loss:  0.25,  Val Acc: 93.14%,  Time: 6:35:40 

第二次测试（随机种子为：2021520）：

最好的效果能够达到0.94左右，但是这个也暂时没办法验证，因为这个是建立在伪标签的基础上的，有些标签并不是正确的

因此，可能与真实结果有些误差，但是相比而言，应该要好一些，毕竟之前用的种子简单分类也是伪标签操作

    Iter:  14700,  Train Loss: 0.0018,  Train Acc: 100.00%,  Val Loss:  0.27,  Val Acc: 94.58%,  Time: 14:56:01 
    Iter:  14800,  Train Loss: 0.0016,  Train Acc: 100.00%,  Val Loss:  0.27,  Val Acc: 94.69%,  Time: 15:02:07 
    Iter:  14900,  Train Loss: 0.0018,  Train Acc: 100.00%,  Val Loss:  0.27,  Val Acc: 94.76%,  Time: 15:08:16 *
    Iter:  15000,  Train Loss: 0.0042,  Train Acc: 100.00%,  Val Loss:  0.27,  Val Acc: 94.73%,  Time: 15:14:22 
    Iter:  15100,  Train Loss: 0.0055,  Train Acc: 100.00%,  Val Loss:  0.26,  Val Acc: 94.71%,  Time: 15:20:25 
