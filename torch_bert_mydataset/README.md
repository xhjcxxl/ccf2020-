# 说明

这个代码是完全重写了dataset，参照着其他代码自己来实现的，很清楚里面的每一个步骤，使得对dataset.dataloader更加深刻

另外，这个代码里面添加了shuffle，所以训练起来更好，更加稳定，之前的代码没有加打乱，训练起来效果非常不好

结构 bert+BiLSTM

结果：ACC:88.46%，应该还可以提升

一次测试结果：

    Iter:   3000,  Train Loss:  0.42,  Train Acc: 75.00%,  Val Loss:  0.46,  Val Acc: 87.88%,  Time: 1:36:44 *