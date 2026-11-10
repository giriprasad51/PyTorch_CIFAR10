import sys
sys.path.insert(0, "/kaggle/working/")


from Additive_Ai.layers import (
    OutputChannelSplitConv2d, 
    InputChannelSplitConv2d, 
    OutputChannelSplitLinear, 
    InputChannelSplitLinear, 
    ParallelActivations, 
    ParallelBatchNorm2d
)

def modify_vgg19(model, num_splits=4):
   
    
    # ==================== BLOCK 1 ====================
    model.features[0] = OutputChannelSplitConv2d(model.features[0], num_splits=num_splits, combine=False)
    model.features[1] = ParallelBatchNorm2d(model.features[1], num_splits=num_splits)
    model.features[2] = ParallelActivations(model.features[2])
    
    model.features[3] = InputChannelSplitConv2d(model.features[3], num_splits=num_splits, combine=True)
    model.features[4] = ParallelActivations(model.features[4])
    model.features[5] = ParallelActivations(model.features[5])
    
    model.features[6] = ParallelActivations(model.features[6], combine=False)

    # ==================== BLOCK 2 ====================
    model.features[7] = OutputChannelSplitConv2d(model.features[7], num_splits=num_splits, combine=False)
    model.features[8] = ParallelBatchNorm2d(model.features[8], num_splits=num_splits)
    model.features[9] = ParallelActivations(model.features[9])
    
    model.features[10] = InputChannelSplitConv2d(model.features[10], num_splits=num_splits, combine=True)
    model.features[11] = ParallelActivations(model.features[11])
    model.features[12] = ParallelActivations(model.features[12])
    
    model.features[13] = ParallelActivations(model.features[13], combine=False)

    # ==================== BLOCK 3 ====================
    model.features[14] = OutputChannelSplitConv2d(model.features[14], num_splits=num_splits, combine=False)
    model.features[15] = ParallelBatchNorm2d(model.features[15], num_splits=num_splits)
    model.features[16] = ParallelActivations(model.features[16])
    
    model.features[17] = InputChannelSplitConv2d(model.features[17], num_splits=num_splits, combine=False)
    model.features[18] = ParallelActivations(model.features[18])
    model.features[19] = ParallelActivations(model.features[19])
    
    model.features[20] = OutputChannelSplitConv2d(model.features[20], num_splits=num_splits, combine=False)
    model.features[21] = ParallelBatchNorm2d(model.features[21], num_splits=num_splits)
    model.features[22] = ParallelActivations(model.features[22])
    
    model.features[23] = InputChannelSplitConv2d(model.features[23], num_splits=num_splits, combine=False)
    model.features[24] = ParallelActivations(model.features[24])
    model.features[25] = ParallelActivations(model.features[25])
    
    model.features[26] = ParallelActivations(model.features[26], combine=False)

    # ==================== BLOCK 4 ====================
    model.features[27] = OutputChannelSplitConv2d(model.features[27], num_splits=num_splits, combine=False)
    model.features[28] = ParallelBatchNorm2d(model.features[28], num_splits=num_splits)
    model.features[29] = ParallelActivations(model.features[29])
    
    model.features[30] = InputChannelSplitConv2d(model.features[30], num_splits=num_splits, combine=False)
    model.features[31] = ParallelActivations(model.features[31])
    model.features[32] = ParallelActivations(model.features[32])
    
    model.features[33] = OutputChannelSplitConv2d(model.features[33], num_splits=num_splits, combine=False)
    model.features[34] = ParallelBatchNorm2d(model.features[34], num_splits=num_splits)
    model.features[35] = ParallelActivations(model.features[35])
    
    model.features[36] = InputChannelSplitConv2d(model.features[36], num_splits=num_splits, combine=True)
    model.features[37] = ParallelActivations(model.features[37])
    model.features[38] = ParallelActivations(model.features[38])
    
    model.features[39] = ParallelActivations(model.features[39], combine=False)

    # ==================== BLOCK 5 ====================
    model.features[40] = OutputChannelSplitConv2d(model.features[40], num_splits=num_splits, combine=False)
    model.features[41] = ParallelBatchNorm2d(model.features[41], num_splits=num_splits)
    model.features[42] = ParallelActivations(model.features[42])
    
    model.features[43] = InputChannelSplitConv2d(model.features[43], num_splits=num_splits, combine=False)
    model.features[44] = ParallelActivations(model.features[44])
    model.features[45] = ParallelActivations(model.features[45])
    
    model.features[46] = OutputChannelSplitConv2d(model.features[46], num_splits=num_splits, combine=False)
    model.features[47] = ParallelBatchNorm2d(model.features[47], num_splits=num_splits)
    model.features[48] = ParallelActivations(model.features[48])
    
    model.features[49] = InputChannelSplitConv2d(model.features[49], num_splits=num_splits, combine=True)
    model.features[50] = ParallelActivations(model.features[50])
    model.features[51] = ParallelActivations(model.features[51])
    
    model.features[52] = ParallelActivations(model.features[52], combine="cat")

    # ==================== AVGPOOL & CLASSIFIER ====================
    model.avgpool = ParallelActivations(activation=model.avgpool, combine=False)
    
    model.classifier[0] = OutputChannelSplitLinear(model.classifier[0], num_splits=num_splits, combine=False)
    model.classifier[1] = ParallelActivations(model.classifier[1])
    model.classifier[2] = ParallelActivations(model.classifier[2])
    model.classifier[3] = InputChannelSplitLinear(model.classifier[3], num_splits=num_splits, combine=True)
    model.classifier[4] = ParallelActivations(model.classifier[4])
    model.classifier[5] = ParallelActivations(model.classifier[5])
    model.classifier[6] = OutputChannelSplitLinear(model.classifier[6], num_splits=num_splits, combine=True)

    return model
