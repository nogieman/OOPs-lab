#pragma once
#define OP_CONV 0x00
// Opcode
#define CONV_Opcode_LOW 0
#define CONV_Opcode_HIGH 3
#define CONV_Opcode_COUNT 4
// Width of the input image
#define CONV_IW_LOW 4
#define CONV_IW_HIGH 13
#define CONV_IW_COUNT 10
// Height of the input image
#define CONV_IH_LOW 14
#define CONV_IH_HIGH 23
#define CONV_IH_COUNT 10
// Channel count for the input
#define CONV_IC_LOW 24
#define CONV_IC_HIGH 35
#define CONV_IC_COUNT 12
// Kernel count for the input
#define CONV_KN_LOW 36
#define CONV_KN_HIGH 47
#define CONV_KN_COUNT 12
// Kernel width
#define CONV_KW_LOW 48
#define CONV_KW_HIGH 51
#define CONV_KW_COUNT 4
// Kernel Height
#define CONV_KH_LOW 52
#define CONV_KH_HIGH 55
#define CONV_KH_COUNT 4
#define CONV_Stride_LOW 56
#define CONV_Stride_HIGH 59
#define CONV_Stride_COUNT 4
#define CONV_PadLeft_LOW 60
#define CONV_PadLeft_HIGH 62
#define CONV_PadLeft_COUNT 3
#define CONV_PadBottom_LOW 63
#define CONV_PadBottom_HIGH 65
#define CONV_PadBottom_COUNT 3
#define CONV_PadRight_LOW 66
#define CONV_PadRight_HIGH 68
#define CONV_PadRight_COUNT 3
#define CONV_PadTop_LOW 69
#define CONV_PadTop_HIGH 71
#define CONV_PadTop_COUNT 3
#define CONV_StartRowSkip_LOW 72
#define CONV_StartRowSkip_HIGH 75
#define CONV_StartRowSkip_COUNT 4
#define CONV_EndRowSkip_LOW 76
#define CONV_EndRowSkip_HIGH 79
#define CONV_EndRowSkip_COUNT 4
#define CONV_ImageStartAddress_LOW 80
#define CONV_ImageStartAddress_HIGH 111
#define CONV_ImageStartAddress_COUNT 32
#define CONV_ImageEndAddress_LOW 112
#define CONV_ImageEndAddress_HIGH 143
#define CONV_ImageEndAddress_COUNT 32
#define CONV_WeightStartAddress_LOW 144
#define CONV_WeightStartAddress_HIGH 175
#define CONV_WeightStartAddress_COUNT 32
#define CONV_WeightEndAddress_LOW 176
#define CONV_WeightEndAddress_HIGH 207
#define CONV_WeightEndAddress_COUNT 32
// Set if the entire image can be fetched in im2col blocks at o
// nce
#define CONV_Im2colPrefetch_LOW 208
#define CONV_Im2colPrefetch_HIGH 208
#define CONV_Im2colPrefetch_COUNT 1
// Channel count for weight
#define CONV_KC_LOW 209
#define CONV_KC_HIGH 220
#define CONV_KC_COUNT 12
#define CONV_ConvType_LOW 221
#define CONV_ConvType_HIGH 222
#define CONV_ConvType_COUNT 2
// If a regular conv is supposed to be performed on a pointwise
// -optimal architecture, this flag is set
#define CONV_ChannelDuplicate_LOW 223
#define CONV_ChannelDuplicate_HIGH 223
#define CONV_ChannelDuplicate_COUNT 1

#define OP_TailBlock 0x01
#define TailBlock_Opcode_LOW 0
#define TailBlock_Opcode_HIGH 3
#define TailBlock_Opcode_COUNT 4
// Batch Norm Yes/No
#define TailBlock_BNEn_LOW 4
#define TailBlock_BNEn_HIGH 4
#define TailBlock_BNEn_COUNT 1
#define TailBlock_BNChannels_LOW 5
#define TailBlock_BNChannels_HIGH 14
#define TailBlock_BNChannels_COUNT 10
#define TailBlock_BNStartAddress_LOW 15
#define TailBlock_BNStartAddress_HIGH 46
#define TailBlock_BNStartAddress_COUNT 32
#define TailBlock_BNEndAddress_LOW 47
#define TailBlock_BNEndAddress_HIGH 78
#define TailBlock_BNEndAddress_COUNT 32
#define TailBlock_ActEn_LOW 79
#define TailBlock_ActEn_HIGH 79
#define TailBlock_ActEn_COUNT 1
#define TailBlock_ActType_LOW 80
#define TailBlock_ActType_HIGH 83
#define TailBlock_ActType_COUNT 4
#define TailBlock_ActParam_LOW 84
#define TailBlock_ActParam_HIGH 91
#define TailBlock_ActParam_COUNT 8
#define TailBlock_QuantEn_LOW 92
#define TailBlock_QuantEn_HIGH 92
#define TailBlock_QuantEn_COUNT 1
#define TailBlock_QuantScale_LOW 93
#define TailBlock_QuantScale_HIGH 108
#define TailBlock_QuantScale_COUNT 16
#define TailBlock_QuantShift_LOW 109
#define TailBlock_QuantShift_HIGH 113
#define TailBlock_QuantShift_COUNT 5
#define TailBlock_PoolEn_LOW 114
#define TailBlock_PoolEn_HIGH 114
#define TailBlock_PoolEn_COUNT 1
#define TailBlock_PoolType_LOW 115
#define TailBlock_PoolType_HIGH 117
#define TailBlock_PoolType_COUNT 3
#define TailBlock_PoolWidth_LOW 118
#define TailBlock_PoolWidth_HIGH 127
#define TailBlock_PoolWidth_COUNT 10
#define TailBlock_PoolHeight_LOW 128
#define TailBlock_PoolHeight_HIGH 137
#define TailBlock_PoolHeight_COUNT 10
#define TailBlock_PoolStride_LOW 138
#define TailBlock_PoolStride_HIGH 141
#define TailBlock_PoolStride_COUNT 4
#define TailBlock_PoolPadding_LOW 142
#define TailBlock_PoolPadding_HIGH 145
#define TailBlock_PoolPadding_COUNT 4
#define TailBlock_PoolCeil_LOW 146
#define TailBlock_PoolCeil_HIGH 146
#define TailBlock_PoolCeil_COUNT 1
// For pools with input size that is not evenly divisible by ke
// rnel size, mod count is the ceil(input % kernel). For exampl
// e, 21x21 for kernel 2x2, mod count is 1 i.e. 1 extra column 
// to be considered.
#define TailBlock_PoolModCount_LOW 147
#define TailBlock_PoolModCount_HIGH 150
#define TailBlock_PoolModCount_COUNT 4
// Same as above but for cols
#define TailBlock_PoolModCountCols_LOW 151
#define TailBlock_PoolModCountCols_HIGH 154
#define TailBlock_PoolModCountCols_COUNT 4
// Same as PadSides for convolution
#define TailBlock_PoolPadSides_LOW 155
#define TailBlock_PoolPadSides_HIGH 158
#define TailBlock_PoolPadSides_COUNT 4
#define TailBlock_BiasEn_LOW 159
#define TailBlock_BiasEn_HIGH 159
#define TailBlock_BiasEn_COUNT 1
// There are two known bias widths 8/32. This is that field.
#define TailBlock_BiasWidth_LOW 160
#define TailBlock_BiasWidth_HIGH 167
#define TailBlock_BiasWidth_COUNT 8
#define TailBlock_BiasStartAddress_LOW 168
#define TailBlock_BiasStartAddress_HIGH 199
#define TailBlock_BiasStartAddress_COUNT 32
#define TailBlock_BiasEndAddress_LOW 200
#define TailBlock_BiasEndAddress_HIGH 231
#define TailBlock_BiasEndAddress_COUNT 32

#define OP_OutputBlock 0x02
#define OutputBlock_Opcode_LOW 0
#define OutputBlock_Opcode_HIGH 3
#define OutputBlock_Opcode_COUNT 4
#define OutputBlock_AccumulantAddr_LOW 4
#define OutputBlock_AccumulantAddr_HIGH 35
#define OutputBlock_AccumulantAddr_COUNT 32
#define OutputBlock_AccumulantReadFirst_LOW 36
#define OutputBlock_AccumulantReadFirst_HIGH 36
#define OutputBlock_AccumulantReadFirst_COUNT 1
#define OutputBlock_OutputAddr_LOW 37
#define OutputBlock_OutputAddr_HIGH 68
#define OutputBlock_OutputAddr_COUNT 32
#define OutputBlock_ChannelItr_LOW 69
#define OutputBlock_ChannelItr_HIGH 80
#define OutputBlock_ChannelItr_COUNT 12
#define OutputBlock_KernelItr_LOW 81
#define OutputBlock_KernelItr_HIGH 92
#define OutputBlock_KernelItr_COUNT 12
// Following the SA, there are tail blocks. Some of the tail bl
// ocks like maxpool modify the shape of the output, this field
//  accounts for that. In cases, when shape is not modified, th
// is field is equal to ImageDimAcc. Additionally, if FlatContr
// oller flag is set to 1, this field is the product of ceil_mo
// d(OC*OH*OW, AXI_WIDTH).
#define OutputBlock_ImageDimOutput_LOW 93
#define OutputBlock_ImageDimOutput_HIGH 108
#define OutputBlock_ImageDimOutput_COUNT 16
// Output of the conv operation (HxW)
#define OutputBlock_ImageDimAcc_LOW 109
#define OutputBlock_ImageDimAcc_HIGH 124
#define OutputBlock_ImageDimAcc_COUNT 16
// For layer with fewer channels than number of columns in the 
// systolic array, accumulation of partial sums across iteratio
// ns is disabled
#define OutputBlock_AccEn_LOW 125
#define OutputBlock_AccEn_HIGH 125
#define OutputBlock_AccEn_COUNT 1
// If this layer's output is supposed to be sent back to the CP
// U, this flag is set
#define OutputBlock_DispatchEn_LOW 126
#define OutputBlock_DispatchEn_HIGH 126
#define OutputBlock_DispatchEn_COUNT 1
// This is a integrity id that the FPGA should attach to the Ad
// dr part of the receiving DWP packet.
#define OutputBlock_DispatchID_LOW 127
#define OutputBlock_DispatchID_HIGH 158
#define OutputBlock_DispatchID_COUNT 32
// If output dimensions of a conv operation can fit on the FPGA
//  output buffers, they should not be sent to the DRAM, all of
//  the conv can happen on chip saving latency. This flag sets 
// that bit.
#define OutputBlock_OnChipAcc_LOW 159
#define OutputBlock_OnChipAcc_HIGH 159
#define OutputBlock_OnChipAcc_COUNT 1
#define OutputBlock_OH_LOW 160
#define OutputBlock_OH_HIGH 169
#define OutputBlock_OH_COUNT 10
#define OutputBlock_OW_LOW 170
#define OutputBlock_OW_HIGH 179
#define OutputBlock_OW_COUNT 10
// If 1, treat outputs from the megablock as flat bytes, not as
//  aligned bytes with zeros in it
#define OutputBlock_FlatController_LOW 180
#define OutputBlock_FlatController_HIGH 180
#define OutputBlock_FlatController_COUNT 1

#define OP_FC 0x03
#define FC_Opcode_LOW 0
#define FC_Opcode_HIGH 3
#define FC_Opcode_COUNT 4
#define FC_WeightRows_LOW 4
#define FC_WeightRows_HIGH 19
#define FC_WeightRows_COUNT 16
#define FC_WeightCols_LOW 20
#define FC_WeightCols_HIGH 35
#define FC_WeightCols_COUNT 16
#define FC_InputRows_LOW 36
#define FC_InputRows_HIGH 51
#define FC_InputRows_COUNT 16
#define FC_DropoutConstant_LOW 52
#define FC_DropoutConstant_HIGH 59
#define FC_DropoutConstant_COUNT 8
// If this FC follows a CONV, the outputs of conv should be fla
// ttened, this bit signals flattening
#define FC_Flatten_LOW 60
#define FC_Flatten_HIGH 60
#define FC_Flatten_COUNT 1
// If flatten is 1, this is the Height x Width of the previous 
// conv. For example, if conv output is 128x7x7, ImageDim will 
// be 49
#define FC_ImageDim_LOW 61
#define FC_ImageDim_HIGH 80
#define FC_ImageDim_COUNT 20
#define FC_ImageStartAddress_LOW 81
#define FC_ImageStartAddress_HIGH 112
#define FC_ImageStartAddress_COUNT 32
#define FC_ImageEndAddr_LOW 113
#define FC_ImageEndAddr_HIGH 144
#define FC_ImageEndAddr_COUNT 32
#define FC_WeightStartAddress_LOW 145
#define FC_WeightStartAddress_HIGH 176
#define FC_WeightStartAddress_COUNT 32
#define FC_WeightEndAddress_LOW 177
#define FC_WeightEndAddress_HIGH 208
#define FC_WeightEndAddress_COUNT 32
// Input vector (say of size 4096) can be seen to be a matrix o
// f size 32x128, vec2mat cols is the number of cols of this ma
// trix i.e. 128
#define FC_Vec2MatCols_LOW 209
#define FC_Vec2MatCols_HIGH 224
#define FC_Vec2MatCols_COUNT 16

#define OP_START 0xf
#define START_Opcode_LOW 0
#define START_Opcode_HIGH 3
#define START_Opcode_COUNT 4
#define START_LayerNumber_LOW 4
#define START_LayerNumber_HIGH 15
#define START_LayerNumber_COUNT 12
#define START_TotalLayers_LOW 16
#define START_TotalLayers_HIGH 27
#define START_TotalLayers_COUNT 12

#define OP_NMS 0x04
// Opcode
#define NMS_Opcode_LOW 0
#define NMS_Opcode_HIGH 3
#define NMS_Opcode_COUNT 4
// IOU Threshold
#define NMS_IOU_LOW 4
#define NMS_IOU_HIGH 19
#define NMS_IOU_COUNT 16
// Shift Value for integer IOU
#define NMS_IOUShift_LOW 20
#define NMS_IOUShift_HIGH 23
#define NMS_IOUShift_COUNT 4
// Score Threshold
#define NMS_ScoreThresh_LOW 24
#define NMS_ScoreThresh_HIGH 39
#define NMS_ScoreThresh_COUNT 16
// Total Boxes in Input
#define NMS_TotalInBoxes_LOW 40
#define NMS_TotalInBoxes_HIGH 59
#define NMS_TotalInBoxes_COUNT 20
// Expected Output Boxes
#define NMS_MaxOutBoxes_LOW 60
#define NMS_MaxOutBoxes_HIGH 67
#define NMS_MaxOutBoxes_COUNT 8
// Whether its ((x1,y1),(x2,y2) or ((h,w),(c1,c2)) (center co-o
// rdinates)
#define NMS_CornerCord_LOW 68
#define NMS_CornerCord_HIGH 68
#define NMS_CornerCord_COUNT 1
// Total Classes in the dataset (for eg., COCO has 80)
#define NMS_TotalClasses_LOW 69
#define NMS_TotalClasses_HIGH 76
#define NMS_TotalClasses_COUNT 8
#define NMS_BoxStartAddr_LOW 77
#define NMS_BoxStartAddr_HIGH 108
#define NMS_BoxStartAddr_COUNT 32
#define NMS_BoxEndAddr_LOW 109
#define NMS_BoxEndAddr_HIGH 140
#define NMS_BoxEndAddr_COUNT 32
#define NMS_ScoreStartAddr_LOW 141
#define NMS_ScoreStartAddr_HIGH 172
#define NMS_ScoreStartAddr_COUNT 32
#define NMS_ScoreEndAddr_LOW 173
#define NMS_ScoreEndAddr_HIGH 204
#define NMS_ScoreEndAddr_COUNT 32

#define OP_EltWise 0x05
// Opcode
#define EltWise_Opcode_LOW 0
#define EltWise_Opcode_HIGH 3
#define EltWise_Opcode_COUNT 4
// Whether its an Add, Sub, Mult etc.
#define EltWise_EltType_LOW 4
#define EltWise_EltType_HIGH 7
#define EltWise_EltType_COUNT 4
#define EltWise_IW_LOW 8
#define EltWise_IW_HIGH 17
#define EltWise_IW_COUNT 10
#define EltWise_IH_LOW 18
#define EltWise_IH_HIGH 27
#define EltWise_IH_COUNT 10
#define EltWise_IC_LOW 28
#define EltWise_IC_HIGH 39
#define EltWise_IC_COUNT 12
#define EltWise_LeftOperandStartAddress_LOW 40
#define EltWise_LeftOperandStartAddress_HIGH 71
#define EltWise_LeftOperandStartAddress_COUNT 32
#define EltWise_LeftOperandEndAddress_LOW 72
#define EltWise_LeftOperandEndAddress_HIGH 103
#define EltWise_LeftOperandEndAddress_COUNT 32
#define EltWise_RightOperandStartAddress_LOW 104
#define EltWise_RightOperandStartAddress_HIGH 135
#define EltWise_RightOperandStartAddress_COUNT 32
#define EltWise_RightOperandEndAddress_LOW 136
#define EltWise_RightOperandEndAddress_HIGH 167
#define EltWise_RightOperandEndAddress_COUNT 32
// FixedPoint32 value of a_scale
#define EltWise_AScale_LOW 168
#define EltWise_AScale_HIGH 199
#define EltWise_AScale_COUNT 32
// FixedPoint32 value of b_scale
#define EltWise_BScale_LOW 200
#define EltWise_BScale_HIGH 231
#define EltWise_BScale_COUNT 32
#define EltWise_AZeroPoint_LOW 232
#define EltWise_AZeroPoint_HIGH 239
#define EltWise_AZeroPoint_COUNT 8
#define EltWise_BZeroPoint_LOW 240
#define EltWise_BZeroPoint_HIGH 247
#define EltWise_BZeroPoint_COUNT 8

#define OP_TRANSPOSE 0x07
#define TRANSPOSE_Opcode_LOW 0
#define TRANSPOSE_Opcode_HIGH 3
#define TRANSPOSE_Opcode_COUNT 4
#define TRANSPOSE_IC_LOW 4
#define TRANSPOSE_IC_HIGH 15
#define TRANSPOSE_IC_COUNT 12
#define TRANSPOSE_IH_LOW 16
#define TRANSPOSE_IH_HIGH 25
#define TRANSPOSE_IH_COUNT 10
#define TRANSPOSE_IW_LOW 26
#define TRANSPOSE_IW_HIGH 35
#define TRANSPOSE_IW_COUNT 10
#define TRANSPOSE_ImageStartAddress_LOW 36
#define TRANSPOSE_ImageStartAddress_HIGH 67
#define TRANSPOSE_ImageStartAddress_COUNT 32

#define OP_RESHAPE 0x06
#define RESHAPE_Opcode_LOW 0
#define RESHAPE_Opcode_HIGH 3
#define RESHAPE_Opcode_COUNT 4
#define RESHAPE_IC_LOW 4
#define RESHAPE_IC_HIGH 15
#define RESHAPE_IC_COUNT 12
#define RESHAPE_IH_LOW 16
#define RESHAPE_IH_HIGH 25
#define RESHAPE_IH_COUNT 10
#define RESHAPE_IW_LOW 26
#define RESHAPE_IW_HIGH 35
#define RESHAPE_IW_COUNT 10
#define RESHAPE_ImageStartAddress_LOW 36
#define RESHAPE_ImageStartAddress_HIGH 67
#define RESHAPE_ImageStartAddress_COUNT 32

#define ISA_VERSION 8
#define ACT_RELU 0x00
#define ACT_CLIP 0x01
#define POOL_MAX 0x00
#define POOL_AVERAGE 0x01
#define POOL_GLOBAL_AVG 0x02
#define WORD_SIZE 32
#define ACC_SIZE 32
#define GATI_INST_ORG 0
#define DWP_HEADER_BYTES 12
#define DWP_PACKET_SZ 4
#define DWP_SOP 0xffffffff
#define DWP_SOP_INDEX 0
#define DWP_DS_INDEX 1
#define DWP_ADDR_INDEX 2
#define META_SOP 0xffffffffffff
#define META_TYPE_RESET 0x000000000000
#define META_TYPE_DISPATCH 0x000000000001
#define META_TYPE_PAYLOAD_SIZE 0x000000000002
#define META_TYPE_INST_ORIGIN 0x000000000003
#define META_CONST_DISPATCH_RAH 0x000000000000
#define META_CONST_DISPATCH_UART 0x000000000001
#define ELTWISE_ADD 0
#define ELTWISE_SUB 1
#define ELTWISE_MULT 2
#define INST_SIZE_BITS 256
#define META_WIDTH_BITS 48
#define RAH_APP_ID 1
#define META_APP_ID 2
#define CONV_TYPE_REGULAR 0
#define CONV_TYPE_DW 1
#define CONV_TYPE_PW 2

#define ZerothStartAddress_LOW 0
#define ZerothStartAddress_HIGH 31
#define ZerothStartAddress_COUNT 32
#define ZerothEndAddress_LOW 32
#define ZerothEndAddress_HIGH 63
#define ZerothEndAddress_COUNT 32


int extract_opcode(const std::bitset<INST_SIZE_BITS> &inst);

template <std::size_t b1N, std::size_t b2N>
unsigned long bitset_range_get(const std::bitset<b2N> &src, int start, int stop);

struct Table {
  std::map<std::string, int> tbl;
  std::vector<std::string> order;
  void clear();
  bool is_empty() const;
};
void print_table(const Table &tbl);

struct pretty_data {
  Table conv;
  Table tailblock;
  Table outputblock;
  Table fc;
  Table start;
  Table nms;
  Table eltwise;
  Table transpose;
  Table reshape;

  void clear() {
    conv.clear();
    tailblock.clear();
    outputblock.clear();
    fc.clear();
    start.clear();
    nms.clear();
    eltwise.clear();
    transpose.clear();
    reshape.clear();
  }
};
inline Table get_conv_table(const std::bitset<INST_SIZE_BITS>& inst) {
	Table tbl;
	tbl.tbl.insert({"Opcode", bitset_range_get<CONV_Opcode_COUNT, INST_SIZE_BITS>(inst, CONV_Opcode_LOW, CONV_Opcode_HIGH)});
	tbl.order.push_back("Opcode");
	tbl.tbl.insert({"IW", bitset_range_get<CONV_IW_COUNT, INST_SIZE_BITS>(inst, CONV_IW_LOW, CONV_IW_HIGH)});
	tbl.order.push_back("IW");
	tbl.tbl.insert({"IH", bitset_range_get<CONV_IH_COUNT, INST_SIZE_BITS>(inst, CONV_IH_LOW, CONV_IH_HIGH)});
	tbl.order.push_back("IH");
	tbl.tbl.insert({"IC", bitset_range_get<CONV_IC_COUNT, INST_SIZE_BITS>(inst, CONV_IC_LOW, CONV_IC_HIGH)});
	tbl.order.push_back("IC");
	tbl.tbl.insert({"KN", bitset_range_get<CONV_KN_COUNT, INST_SIZE_BITS>(inst, CONV_KN_LOW, CONV_KN_HIGH)});
	tbl.order.push_back("KN");
	tbl.tbl.insert({"KW", bitset_range_get<CONV_KW_COUNT, INST_SIZE_BITS>(inst, CONV_KW_LOW, CONV_KW_HIGH)});
	tbl.order.push_back("KW");
	tbl.tbl.insert({"KH", bitset_range_get<CONV_KH_COUNT, INST_SIZE_BITS>(inst, CONV_KH_LOW, CONV_KH_HIGH)});
	tbl.order.push_back("KH");
	tbl.tbl.insert({"Stride", bitset_range_get<CONV_Stride_COUNT, INST_SIZE_BITS>(inst, CONV_Stride_LOW, CONV_Stride_HIGH)});
	tbl.order.push_back("Stride");
	tbl.tbl.insert({"PadLeft", bitset_range_get<CONV_PadLeft_COUNT, INST_SIZE_BITS>(inst, CONV_PadLeft_LOW, CONV_PadLeft_HIGH)});
	tbl.order.push_back("PadLeft");
	tbl.tbl.insert({"PadBottom", bitset_range_get<CONV_PadBottom_COUNT, INST_SIZE_BITS>(inst, CONV_PadBottom_LOW, CONV_PadBottom_HIGH)});
	tbl.order.push_back("PadBottom");
	tbl.tbl.insert({"PadRight", bitset_range_get<CONV_PadRight_COUNT, INST_SIZE_BITS>(inst, CONV_PadRight_LOW, CONV_PadRight_HIGH)});
	tbl.order.push_back("PadRight");
	tbl.tbl.insert({"PadTop", bitset_range_get<CONV_PadTop_COUNT, INST_SIZE_BITS>(inst, CONV_PadTop_LOW, CONV_PadTop_HIGH)});
	tbl.order.push_back("PadTop");
	tbl.tbl.insert({"StartRowSkip", bitset_range_get<CONV_StartRowSkip_COUNT, INST_SIZE_BITS>(inst, CONV_StartRowSkip_LOW, CONV_StartRowSkip_HIGH)});
	tbl.order.push_back("StartRowSkip");
	tbl.tbl.insert({"EndRowSkip", bitset_range_get<CONV_EndRowSkip_COUNT, INST_SIZE_BITS>(inst, CONV_EndRowSkip_LOW, CONV_EndRowSkip_HIGH)});
	tbl.order.push_back("EndRowSkip");
	tbl.tbl.insert({"ImageStartAddress", bitset_range_get<CONV_ImageStartAddress_COUNT, INST_SIZE_BITS>(inst, CONV_ImageStartAddress_LOW, CONV_ImageStartAddress_HIGH)});
	tbl.order.push_back("ImageStartAddress");
	tbl.tbl.insert({"ImageEndAddress", bitset_range_get<CONV_ImageEndAddress_COUNT, INST_SIZE_BITS>(inst, CONV_ImageEndAddress_LOW, CONV_ImageEndAddress_HIGH)});
	tbl.order.push_back("ImageEndAddress");
	tbl.tbl.insert({"WeightStartAddress", bitset_range_get<CONV_WeightStartAddress_COUNT, INST_SIZE_BITS>(inst, CONV_WeightStartAddress_LOW, CONV_WeightStartAddress_HIGH)});
	tbl.order.push_back("WeightStartAddress");
	tbl.tbl.insert({"WeightEndAddress", bitset_range_get<CONV_WeightEndAddress_COUNT, INST_SIZE_BITS>(inst, CONV_WeightEndAddress_LOW, CONV_WeightEndAddress_HIGH)});
	tbl.order.push_back("WeightEndAddress");
	tbl.tbl.insert({"Im2colPrefetch", bitset_range_get<CONV_Im2colPrefetch_COUNT, INST_SIZE_BITS>(inst, CONV_Im2colPrefetch_LOW, CONV_Im2colPrefetch_HIGH)});
	tbl.order.push_back("Im2colPrefetch");
	tbl.tbl.insert({"KC", bitset_range_get<CONV_KC_COUNT, INST_SIZE_BITS>(inst, CONV_KC_LOW, CONV_KC_HIGH)});
	tbl.order.push_back("KC");
	tbl.tbl.insert({"ConvType", bitset_range_get<CONV_ConvType_COUNT, INST_SIZE_BITS>(inst, CONV_ConvType_LOW, CONV_ConvType_HIGH)});
	tbl.order.push_back("ConvType");
	tbl.tbl.insert({"ChannelDuplicate", bitset_range_get<CONV_ChannelDuplicate_COUNT, INST_SIZE_BITS>(inst, CONV_ChannelDuplicate_LOW, CONV_ChannelDuplicate_HIGH)});
	tbl.order.push_back("ChannelDuplicate");
	return tbl;
}
inline void pretty_print_conv(const std::bitset<INST_SIZE_BITS>& inst) {
	auto tbl = get_conv_table(inst);
	print_table(tbl);
}
inline Table get_tailblock_table(const std::bitset<INST_SIZE_BITS>& inst) {
	Table tbl;
	tbl.tbl.insert({"Opcode", bitset_range_get<TailBlock_Opcode_COUNT, INST_SIZE_BITS>(inst, TailBlock_Opcode_LOW, TailBlock_Opcode_HIGH)});
	tbl.order.push_back("Opcode");
	tbl.tbl.insert({"BNEn", bitset_range_get<TailBlock_BNEn_COUNT, INST_SIZE_BITS>(inst, TailBlock_BNEn_LOW, TailBlock_BNEn_HIGH)});
	tbl.order.push_back("BNEn");
	tbl.tbl.insert({"BNChannels", bitset_range_get<TailBlock_BNChannels_COUNT, INST_SIZE_BITS>(inst, TailBlock_BNChannels_LOW, TailBlock_BNChannels_HIGH)});
	tbl.order.push_back("BNChannels");
	tbl.tbl.insert({"BNStartAddress", bitset_range_get<TailBlock_BNStartAddress_COUNT, INST_SIZE_BITS>(inst, TailBlock_BNStartAddress_LOW, TailBlock_BNStartAddress_HIGH)});
	tbl.order.push_back("BNStartAddress");
	tbl.tbl.insert({"BNEndAddress", bitset_range_get<TailBlock_BNEndAddress_COUNT, INST_SIZE_BITS>(inst, TailBlock_BNEndAddress_LOW, TailBlock_BNEndAddress_HIGH)});
	tbl.order.push_back("BNEndAddress");
	tbl.tbl.insert({"ActEn", bitset_range_get<TailBlock_ActEn_COUNT, INST_SIZE_BITS>(inst, TailBlock_ActEn_LOW, TailBlock_ActEn_HIGH)});
	tbl.order.push_back("ActEn");
	tbl.tbl.insert({"ActType", bitset_range_get<TailBlock_ActType_COUNT, INST_SIZE_BITS>(inst, TailBlock_ActType_LOW, TailBlock_ActType_HIGH)});
	tbl.order.push_back("ActType");
	tbl.tbl.insert({"ActParam", bitset_range_get<TailBlock_ActParam_COUNT, INST_SIZE_BITS>(inst, TailBlock_ActParam_LOW, TailBlock_ActParam_HIGH)});
	tbl.order.push_back("ActParam");
	tbl.tbl.insert({"QuantEn", bitset_range_get<TailBlock_QuantEn_COUNT, INST_SIZE_BITS>(inst, TailBlock_QuantEn_LOW, TailBlock_QuantEn_HIGH)});
	tbl.order.push_back("QuantEn");
	tbl.tbl.insert({"QuantScale", bitset_range_get<TailBlock_QuantScale_COUNT, INST_SIZE_BITS>(inst, TailBlock_QuantScale_LOW, TailBlock_QuantScale_HIGH)});
	tbl.order.push_back("QuantScale");
	tbl.tbl.insert({"QuantShift", bitset_range_get<TailBlock_QuantShift_COUNT, INST_SIZE_BITS>(inst, TailBlock_QuantShift_LOW, TailBlock_QuantShift_HIGH)});
	tbl.order.push_back("QuantShift");
	tbl.tbl.insert({"PoolEn", bitset_range_get<TailBlock_PoolEn_COUNT, INST_SIZE_BITS>(inst, TailBlock_PoolEn_LOW, TailBlock_PoolEn_HIGH)});
	tbl.order.push_back("PoolEn");
	tbl.tbl.insert({"PoolType", bitset_range_get<TailBlock_PoolType_COUNT, INST_SIZE_BITS>(inst, TailBlock_PoolType_LOW, TailBlock_PoolType_HIGH)});
	tbl.order.push_back("PoolType");
	tbl.tbl.insert({"PoolWidth", bitset_range_get<TailBlock_PoolWidth_COUNT, INST_SIZE_BITS>(inst, TailBlock_PoolWidth_LOW, TailBlock_PoolWidth_HIGH)});
	tbl.order.push_back("PoolWidth");
	tbl.tbl.insert({"PoolHeight", bitset_range_get<TailBlock_PoolHeight_COUNT, INST_SIZE_BITS>(inst, TailBlock_PoolHeight_LOW, TailBlock_PoolHeight_HIGH)});
	tbl.order.push_back("PoolHeight");
	tbl.tbl.insert({"PoolStride", bitset_range_get<TailBlock_PoolStride_COUNT, INST_SIZE_BITS>(inst, TailBlock_PoolStride_LOW, TailBlock_PoolStride_HIGH)});
	tbl.order.push_back("PoolStride");
	tbl.tbl.insert({"PoolPadding", bitset_range_get<TailBlock_PoolPadding_COUNT, INST_SIZE_BITS>(inst, TailBlock_PoolPadding_LOW, TailBlock_PoolPadding_HIGH)});
	tbl.order.push_back("PoolPadding");
	tbl.tbl.insert({"PoolCeil", bitset_range_get<TailBlock_PoolCeil_COUNT, INST_SIZE_BITS>(inst, TailBlock_PoolCeil_LOW, TailBlock_PoolCeil_HIGH)});
	tbl.order.push_back("PoolCeil");
	tbl.tbl.insert({"PoolModCount", bitset_range_get<TailBlock_PoolModCount_COUNT, INST_SIZE_BITS>(inst, TailBlock_PoolModCount_LOW, TailBlock_PoolModCount_HIGH)});
	tbl.order.push_back("PoolModCount");
	tbl.tbl.insert({"PoolModCountCols", bitset_range_get<TailBlock_PoolModCountCols_COUNT, INST_SIZE_BITS>(inst, TailBlock_PoolModCountCols_LOW, TailBlock_PoolModCountCols_HIGH)});
	tbl.order.push_back("PoolModCountCols");
	tbl.tbl.insert({"PoolPadSides", bitset_range_get<TailBlock_PoolPadSides_COUNT, INST_SIZE_BITS>(inst, TailBlock_PoolPadSides_LOW, TailBlock_PoolPadSides_HIGH)});
	tbl.order.push_back("PoolPadSides");
	tbl.tbl.insert({"BiasEn", bitset_range_get<TailBlock_BiasEn_COUNT, INST_SIZE_BITS>(inst, TailBlock_BiasEn_LOW, TailBlock_BiasEn_HIGH)});
	tbl.order.push_back("BiasEn");
	tbl.tbl.insert({"BiasWidth", bitset_range_get<TailBlock_BiasWidth_COUNT, INST_SIZE_BITS>(inst, TailBlock_BiasWidth_LOW, TailBlock_BiasWidth_HIGH)});
	tbl.order.push_back("BiasWidth");
	tbl.tbl.insert({"BiasStartAddress", bitset_range_get<TailBlock_BiasStartAddress_COUNT, INST_SIZE_BITS>(inst, TailBlock_BiasStartAddress_LOW, TailBlock_BiasStartAddress_HIGH)});
	tbl.order.push_back("BiasStartAddress");
	tbl.tbl.insert({"BiasEndAddress", bitset_range_get<TailBlock_BiasEndAddress_COUNT, INST_SIZE_BITS>(inst, TailBlock_BiasEndAddress_LOW, TailBlock_BiasEndAddress_HIGH)});
	tbl.order.push_back("BiasEndAddress");
	return tbl;
}
inline void pretty_print_tailblock(const std::bitset<INST_SIZE_BITS>& inst) {
	auto tbl = get_tailblock_table(inst);
	print_table(tbl);
}
inline Table get_outputblock_table(const std::bitset<INST_SIZE_BITS>& inst) {
	Table tbl;
	tbl.tbl.insert({"Opcode", bitset_range_get<OutputBlock_Opcode_COUNT, INST_SIZE_BITS>(inst, OutputBlock_Opcode_LOW, OutputBlock_Opcode_HIGH)});
	tbl.order.push_back("Opcode");
	tbl.tbl.insert({"AccumulantAddr", bitset_range_get<OutputBlock_AccumulantAddr_COUNT, INST_SIZE_BITS>(inst, OutputBlock_AccumulantAddr_LOW, OutputBlock_AccumulantAddr_HIGH)});
	tbl.order.push_back("AccumulantAddr");
	tbl.tbl.insert({"AccumulantReadFirst", bitset_range_get<OutputBlock_AccumulantReadFirst_COUNT, INST_SIZE_BITS>(inst, OutputBlock_AccumulantReadFirst_LOW, OutputBlock_AccumulantReadFirst_HIGH)});
	tbl.order.push_back("AccumulantReadFirst");
	tbl.tbl.insert({"OutputAddr", bitset_range_get<OutputBlock_OutputAddr_COUNT, INST_SIZE_BITS>(inst, OutputBlock_OutputAddr_LOW, OutputBlock_OutputAddr_HIGH)});
	tbl.order.push_back("OutputAddr");
	tbl.tbl.insert({"ChannelItr", bitset_range_get<OutputBlock_ChannelItr_COUNT, INST_SIZE_BITS>(inst, OutputBlock_ChannelItr_LOW, OutputBlock_ChannelItr_HIGH)});
	tbl.order.push_back("ChannelItr");
	tbl.tbl.insert({"KernelItr", bitset_range_get<OutputBlock_KernelItr_COUNT, INST_SIZE_BITS>(inst, OutputBlock_KernelItr_LOW, OutputBlock_KernelItr_HIGH)});
	tbl.order.push_back("KernelItr");
	tbl.tbl.insert({"ImageDimOutput", bitset_range_get<OutputBlock_ImageDimOutput_COUNT, INST_SIZE_BITS>(inst, OutputBlock_ImageDimOutput_LOW, OutputBlock_ImageDimOutput_HIGH)});
	tbl.order.push_back("ImageDimOutput");
	tbl.tbl.insert({"ImageDimAcc", bitset_range_get<OutputBlock_ImageDimAcc_COUNT, INST_SIZE_BITS>(inst, OutputBlock_ImageDimAcc_LOW, OutputBlock_ImageDimAcc_HIGH)});
	tbl.order.push_back("ImageDimAcc");
	tbl.tbl.insert({"AccEn", bitset_range_get<OutputBlock_AccEn_COUNT, INST_SIZE_BITS>(inst, OutputBlock_AccEn_LOW, OutputBlock_AccEn_HIGH)});
	tbl.order.push_back("AccEn");
	tbl.tbl.insert({"DispatchEn", bitset_range_get<OutputBlock_DispatchEn_COUNT, INST_SIZE_BITS>(inst, OutputBlock_DispatchEn_LOW, OutputBlock_DispatchEn_HIGH)});
	tbl.order.push_back("DispatchEn");
	tbl.tbl.insert({"DispatchID", bitset_range_get<OutputBlock_DispatchID_COUNT, INST_SIZE_BITS>(inst, OutputBlock_DispatchID_LOW, OutputBlock_DispatchID_HIGH)});
	tbl.order.push_back("DispatchID");
	tbl.tbl.insert({"OnChipAcc", bitset_range_get<OutputBlock_OnChipAcc_COUNT, INST_SIZE_BITS>(inst, OutputBlock_OnChipAcc_LOW, OutputBlock_OnChipAcc_HIGH)});
	tbl.order.push_back("OnChipAcc");
	tbl.tbl.insert({"OH", bitset_range_get<OutputBlock_OH_COUNT, INST_SIZE_BITS>(inst, OutputBlock_OH_LOW, OutputBlock_OH_HIGH)});
	tbl.order.push_back("OH");
	tbl.tbl.insert({"OW", bitset_range_get<OutputBlock_OW_COUNT, INST_SIZE_BITS>(inst, OutputBlock_OW_LOW, OutputBlock_OW_HIGH)});
	tbl.order.push_back("OW");
	tbl.tbl.insert({"FlatController", bitset_range_get<OutputBlock_FlatController_COUNT, INST_SIZE_BITS>(inst, OutputBlock_FlatController_LOW, OutputBlock_FlatController_HIGH)});
	tbl.order.push_back("FlatController");
	return tbl;
}
inline void pretty_print_outputblock(const std::bitset<INST_SIZE_BITS>& inst) {
	auto tbl = get_outputblock_table(inst);
	print_table(tbl);
}
inline Table get_fc_table(const std::bitset<INST_SIZE_BITS>& inst) {
	Table tbl;
	tbl.tbl.insert({"Opcode", bitset_range_get<FC_Opcode_COUNT, INST_SIZE_BITS>(inst, FC_Opcode_LOW, FC_Opcode_HIGH)});
	tbl.order.push_back("Opcode");
	tbl.tbl.insert({"WeightRows", bitset_range_get<FC_WeightRows_COUNT, INST_SIZE_BITS>(inst, FC_WeightRows_LOW, FC_WeightRows_HIGH)});
	tbl.order.push_back("WeightRows");
	tbl.tbl.insert({"WeightCols", bitset_range_get<FC_WeightCols_COUNT, INST_SIZE_BITS>(inst, FC_WeightCols_LOW, FC_WeightCols_HIGH)});
	tbl.order.push_back("WeightCols");
	tbl.tbl.insert({"InputRows", bitset_range_get<FC_InputRows_COUNT, INST_SIZE_BITS>(inst, FC_InputRows_LOW, FC_InputRows_HIGH)});
	tbl.order.push_back("InputRows");
	tbl.tbl.insert({"DropoutConstant", bitset_range_get<FC_DropoutConstant_COUNT, INST_SIZE_BITS>(inst, FC_DropoutConstant_LOW, FC_DropoutConstant_HIGH)});
	tbl.order.push_back("DropoutConstant");
	tbl.tbl.insert({"Flatten", bitset_range_get<FC_Flatten_COUNT, INST_SIZE_BITS>(inst, FC_Flatten_LOW, FC_Flatten_HIGH)});
	tbl.order.push_back("Flatten");
	tbl.tbl.insert({"ImageDim", bitset_range_get<FC_ImageDim_COUNT, INST_SIZE_BITS>(inst, FC_ImageDim_LOW, FC_ImageDim_HIGH)});
	tbl.order.push_back("ImageDim");
	tbl.tbl.insert({"ImageStartAddress", bitset_range_get<FC_ImageStartAddress_COUNT, INST_SIZE_BITS>(inst, FC_ImageStartAddress_LOW, FC_ImageStartAddress_HIGH)});
	tbl.order.push_back("ImageStartAddress");
	tbl.tbl.insert({"ImageEndAddr", bitset_range_get<FC_ImageEndAddr_COUNT, INST_SIZE_BITS>(inst, FC_ImageEndAddr_LOW, FC_ImageEndAddr_HIGH)});
	tbl.order.push_back("ImageEndAddr");
	tbl.tbl.insert({"WeightStartAddress", bitset_range_get<FC_WeightStartAddress_COUNT, INST_SIZE_BITS>(inst, FC_WeightStartAddress_LOW, FC_WeightStartAddress_HIGH)});
	tbl.order.push_back("WeightStartAddress");
	tbl.tbl.insert({"WeightEndAddress", bitset_range_get<FC_WeightEndAddress_COUNT, INST_SIZE_BITS>(inst, FC_WeightEndAddress_LOW, FC_WeightEndAddress_HIGH)});
	tbl.order.push_back("WeightEndAddress");
	tbl.tbl.insert({"Vec2MatCols", bitset_range_get<FC_Vec2MatCols_COUNT, INST_SIZE_BITS>(inst, FC_Vec2MatCols_LOW, FC_Vec2MatCols_HIGH)});
	tbl.order.push_back("Vec2MatCols");
	return tbl;
}
inline void pretty_print_fc(const std::bitset<INST_SIZE_BITS>& inst) {
	auto tbl = get_fc_table(inst);
	print_table(tbl);
}
inline Table get_start_table(const std::bitset<INST_SIZE_BITS>& inst) {
	Table tbl;
	tbl.tbl.insert({"Opcode", bitset_range_get<START_Opcode_COUNT, INST_SIZE_BITS>(inst, START_Opcode_LOW, START_Opcode_HIGH)});
	tbl.order.push_back("Opcode");
	tbl.tbl.insert({"LayerNumber", bitset_range_get<START_LayerNumber_COUNT, INST_SIZE_BITS>(inst, START_LayerNumber_LOW, START_LayerNumber_HIGH)});
	tbl.order.push_back("LayerNumber");
	tbl.tbl.insert({"TotalLayers", bitset_range_get<START_TotalLayers_COUNT, INST_SIZE_BITS>(inst, START_TotalLayers_LOW, START_TotalLayers_HIGH)});
	tbl.order.push_back("TotalLayers");
	return tbl;
}
inline void pretty_print_start(const std::bitset<INST_SIZE_BITS>& inst) {
	auto tbl = get_start_table(inst);
	print_table(tbl);
}
inline Table get_nms_table(const std::bitset<INST_SIZE_BITS>& inst) {
	Table tbl;
	tbl.tbl.insert({"Opcode", bitset_range_get<NMS_Opcode_COUNT, INST_SIZE_BITS>(inst, NMS_Opcode_LOW, NMS_Opcode_HIGH)});
	tbl.order.push_back("Opcode");
	tbl.tbl.insert({"IOU", bitset_range_get<NMS_IOU_COUNT, INST_SIZE_BITS>(inst, NMS_IOU_LOW, NMS_IOU_HIGH)});
	tbl.order.push_back("IOU");
	tbl.tbl.insert({"IOUShift", bitset_range_get<NMS_IOUShift_COUNT, INST_SIZE_BITS>(inst, NMS_IOUShift_LOW, NMS_IOUShift_HIGH)});
	tbl.order.push_back("IOUShift");
	tbl.tbl.insert({"ScoreThresh", bitset_range_get<NMS_ScoreThresh_COUNT, INST_SIZE_BITS>(inst, NMS_ScoreThresh_LOW, NMS_ScoreThresh_HIGH)});
	tbl.order.push_back("ScoreThresh");
	tbl.tbl.insert({"TotalInBoxes", bitset_range_get<NMS_TotalInBoxes_COUNT, INST_SIZE_BITS>(inst, NMS_TotalInBoxes_LOW, NMS_TotalInBoxes_HIGH)});
	tbl.order.push_back("TotalInBoxes");
	tbl.tbl.insert({"MaxOutBoxes", bitset_range_get<NMS_MaxOutBoxes_COUNT, INST_SIZE_BITS>(inst, NMS_MaxOutBoxes_LOW, NMS_MaxOutBoxes_HIGH)});
	tbl.order.push_back("MaxOutBoxes");
	tbl.tbl.insert({"CornerCord", bitset_range_get<NMS_CornerCord_COUNT, INST_SIZE_BITS>(inst, NMS_CornerCord_LOW, NMS_CornerCord_HIGH)});
	tbl.order.push_back("CornerCord");
	tbl.tbl.insert({"TotalClasses", bitset_range_get<NMS_TotalClasses_COUNT, INST_SIZE_BITS>(inst, NMS_TotalClasses_LOW, NMS_TotalClasses_HIGH)});
	tbl.order.push_back("TotalClasses");
	tbl.tbl.insert({"BoxStartAddr", bitset_range_get<NMS_BoxStartAddr_COUNT, INST_SIZE_BITS>(inst, NMS_BoxStartAddr_LOW, NMS_BoxStartAddr_HIGH)});
	tbl.order.push_back("BoxStartAddr");
	tbl.tbl.insert({"BoxEndAddr", bitset_range_get<NMS_BoxEndAddr_COUNT, INST_SIZE_BITS>(inst, NMS_BoxEndAddr_LOW, NMS_BoxEndAddr_HIGH)});
	tbl.order.push_back("BoxEndAddr");
	tbl.tbl.insert({"ScoreStartAddr", bitset_range_get<NMS_ScoreStartAddr_COUNT, INST_SIZE_BITS>(inst, NMS_ScoreStartAddr_LOW, NMS_ScoreStartAddr_HIGH)});
	tbl.order.push_back("ScoreStartAddr");
	tbl.tbl.insert({"ScoreEndAddr", bitset_range_get<NMS_ScoreEndAddr_COUNT, INST_SIZE_BITS>(inst, NMS_ScoreEndAddr_LOW, NMS_ScoreEndAddr_HIGH)});
	tbl.order.push_back("ScoreEndAddr");
	return tbl;
}
inline void pretty_print_nms(const std::bitset<INST_SIZE_BITS>& inst) {
	auto tbl = get_nms_table(inst);
	print_table(tbl);
}
inline Table get_eltwise_table(const std::bitset<INST_SIZE_BITS>& inst) {
	Table tbl;
	tbl.tbl.insert({"Opcode", bitset_range_get<EltWise_Opcode_COUNT, INST_SIZE_BITS>(inst, EltWise_Opcode_LOW, EltWise_Opcode_HIGH)});
	tbl.order.push_back("Opcode");
	tbl.tbl.insert({"EltType", bitset_range_get<EltWise_EltType_COUNT, INST_SIZE_BITS>(inst, EltWise_EltType_LOW, EltWise_EltType_HIGH)});
	tbl.order.push_back("EltType");
	tbl.tbl.insert({"IW", bitset_range_get<EltWise_IW_COUNT, INST_SIZE_BITS>(inst, EltWise_IW_LOW, EltWise_IW_HIGH)});
	tbl.order.push_back("IW");
	tbl.tbl.insert({"IH", bitset_range_get<EltWise_IH_COUNT, INST_SIZE_BITS>(inst, EltWise_IH_LOW, EltWise_IH_HIGH)});
	tbl.order.push_back("IH");
	tbl.tbl.insert({"IC", bitset_range_get<EltWise_IC_COUNT, INST_SIZE_BITS>(inst, EltWise_IC_LOW, EltWise_IC_HIGH)});
	tbl.order.push_back("IC");
	tbl.tbl.insert({"LeftOperandStartAddress", bitset_range_get<EltWise_LeftOperandStartAddress_COUNT, INST_SIZE_BITS>(inst, EltWise_LeftOperandStartAddress_LOW, EltWise_LeftOperandStartAddress_HIGH)});
	tbl.order.push_back("LeftOperandStartAddress");
	tbl.tbl.insert({"LeftOperandEndAddress", bitset_range_get<EltWise_LeftOperandEndAddress_COUNT, INST_SIZE_BITS>(inst, EltWise_LeftOperandEndAddress_LOW, EltWise_LeftOperandEndAddress_HIGH)});
	tbl.order.push_back("LeftOperandEndAddress");
	tbl.tbl.insert({"RightOperandStartAddress", bitset_range_get<EltWise_RightOperandStartAddress_COUNT, INST_SIZE_BITS>(inst, EltWise_RightOperandStartAddress_LOW, EltWise_RightOperandStartAddress_HIGH)});
	tbl.order.push_back("RightOperandStartAddress");
	tbl.tbl.insert({"RightOperandEndAddress", bitset_range_get<EltWise_RightOperandEndAddress_COUNT, INST_SIZE_BITS>(inst, EltWise_RightOperandEndAddress_LOW, EltWise_RightOperandEndAddress_HIGH)});
	tbl.order.push_back("RightOperandEndAddress");
	tbl.tbl.insert({"AScale", bitset_range_get<EltWise_AScale_COUNT, INST_SIZE_BITS>(inst, EltWise_AScale_LOW, EltWise_AScale_HIGH)});
	tbl.order.push_back("AScale");
	tbl.tbl.insert({"BScale", bitset_range_get<EltWise_BScale_COUNT, INST_SIZE_BITS>(inst, EltWise_BScale_LOW, EltWise_BScale_HIGH)});
	tbl.order.push_back("BScale");
	tbl.tbl.insert({"AZeroPoint", bitset_range_get<EltWise_AZeroPoint_COUNT, INST_SIZE_BITS>(inst, EltWise_AZeroPoint_LOW, EltWise_AZeroPoint_HIGH)});
	tbl.order.push_back("AZeroPoint");
	tbl.tbl.insert({"BZeroPoint", bitset_range_get<EltWise_BZeroPoint_COUNT, INST_SIZE_BITS>(inst, EltWise_BZeroPoint_LOW, EltWise_BZeroPoint_HIGH)});
	tbl.order.push_back("BZeroPoint");
	return tbl;
}
inline void pretty_print_eltwise(const std::bitset<INST_SIZE_BITS>& inst) {
	auto tbl = get_eltwise_table(inst);
	print_table(tbl);
}
inline Table get_transpose_table(const std::bitset<INST_SIZE_BITS>& inst) {
	Table tbl;
	tbl.tbl.insert({"Opcode", bitset_range_get<TRANSPOSE_Opcode_COUNT, INST_SIZE_BITS>(inst, TRANSPOSE_Opcode_LOW, TRANSPOSE_Opcode_HIGH)});
	tbl.order.push_back("Opcode");
	tbl.tbl.insert({"IC", bitset_range_get<TRANSPOSE_IC_COUNT, INST_SIZE_BITS>(inst, TRANSPOSE_IC_LOW, TRANSPOSE_IC_HIGH)});
	tbl.order.push_back("IC");
	tbl.tbl.insert({"IH", bitset_range_get<TRANSPOSE_IH_COUNT, INST_SIZE_BITS>(inst, TRANSPOSE_IH_LOW, TRANSPOSE_IH_HIGH)});
	tbl.order.push_back("IH");
	tbl.tbl.insert({"IW", bitset_range_get<TRANSPOSE_IW_COUNT, INST_SIZE_BITS>(inst, TRANSPOSE_IW_LOW, TRANSPOSE_IW_HIGH)});
	tbl.order.push_back("IW");
	tbl.tbl.insert({"ImageStartAddress", bitset_range_get<TRANSPOSE_ImageStartAddress_COUNT, INST_SIZE_BITS>(inst, TRANSPOSE_ImageStartAddress_LOW, TRANSPOSE_ImageStartAddress_HIGH)});
	tbl.order.push_back("ImageStartAddress");
	return tbl;
}
inline void pretty_print_transpose(const std::bitset<INST_SIZE_BITS>& inst) {
	auto tbl = get_transpose_table(inst);
	print_table(tbl);
}
inline Table get_reshape_table(const std::bitset<INST_SIZE_BITS>& inst) {
	Table tbl;
	tbl.tbl.insert({"Opcode", bitset_range_get<RESHAPE_Opcode_COUNT, INST_SIZE_BITS>(inst, RESHAPE_Opcode_LOW, RESHAPE_Opcode_HIGH)});
	tbl.order.push_back("Opcode");
	tbl.tbl.insert({"IC", bitset_range_get<RESHAPE_IC_COUNT, INST_SIZE_BITS>(inst, RESHAPE_IC_LOW, RESHAPE_IC_HIGH)});
	tbl.order.push_back("IC");
	tbl.tbl.insert({"IH", bitset_range_get<RESHAPE_IH_COUNT, INST_SIZE_BITS>(inst, RESHAPE_IH_LOW, RESHAPE_IH_HIGH)});
	tbl.order.push_back("IH");
	tbl.tbl.insert({"IW", bitset_range_get<RESHAPE_IW_COUNT, INST_SIZE_BITS>(inst, RESHAPE_IW_LOW, RESHAPE_IW_HIGH)});
	tbl.order.push_back("IW");
	tbl.tbl.insert({"ImageStartAddress", bitset_range_get<RESHAPE_ImageStartAddress_COUNT, INST_SIZE_BITS>(inst, RESHAPE_ImageStartAddress_LOW, RESHAPE_ImageStartAddress_HIGH)});
	tbl.order.push_back("ImageStartAddress");
	return tbl;
}
inline void pretty_print_reshape(const std::bitset<INST_SIZE_BITS>& inst) {
	auto tbl = get_reshape_table(inst);
	print_table(tbl);
}

inline void pretty_print(const std::bitset<INST_SIZE_BITS> &inst) {
  int op_code = extract_opcode(inst);
  switch (op_code) {
  case OP_CONV:
    pretty_print_conv(inst);
    break;
  case OP_TailBlock:
    pretty_print_tailblock(inst);
    break;
  case OP_OutputBlock:
    pretty_print_outputblock(inst);
    break;
  case OP_FC:
    pretty_print_fc(inst);
    break;
  case OP_START:
    pretty_print_start(inst);
    break;
  case OP_NMS:
    pretty_print_nms(inst);
    break;
  case OP_EltWise:
    pretty_print_eltwise(inst);
    break;
  case OP_TRANSPOSE:
    pretty_print_transpose(inst);
    break;
  case OP_RESHAPE:
    pretty_print_reshape(inst);
    break;
  default:
    log_fatal("can't pretty print instruction with opcode {}\n", op_code);
    break;
  }
}


inline void pretty_print_html(const std::bitset<INST_SIZE_BITS> &inst,
                              std::vector<pretty_data> &data,
                              pretty_data &inst_data) {
  int op_code = extract_opcode(inst);
  switch (op_code) {
  case OP_CONV:
    inst_data.conv = get_conv_table(inst);
    break;
  case OP_TailBlock:
    inst_data.tailblock = get_tailblock_table(inst);
    break;
  case OP_OutputBlock:
    inst_data.outputblock = get_outputblock_table(inst);
    break;
  case OP_FC:
    inst_data.fc = get_fc_table(inst);
    break;
  case OP_START:
    inst_data.start = get_start_table(inst);
    data.push_back(inst_data);
    inst_data.clear();
    break;
  case OP_NMS:
    inst_data.nms = get_nms_table(inst);
    break;
  case OP_EltWise:
    inst_data.eltwise = get_eltwise_table(inst);
    break;
  case OP_TRANSPOSE:
    inst_data.transpose = get_transpose_table(inst);
    break;
  case OP_RESHAPE:
    inst_data.reshape = get_reshape_table(inst);
    break;
  default:
    log_fatal("can't pretty print instruction with opcode {}\n", op_code);
    break;
  }
}

inline std::string generate_table_html(const std::string &tableName,
                                       const Table &table) {
  if (table.is_empty())
    return "";

  std::ostringstream html;
  html << "<div class='collapsible'>" << tableName << "</div>\n";
  html << "<div class='content'>\n";
  html << "<ul>\n";

  for (const auto &key : table.order) {
    if (table.tbl.count(key)) {
      html << "<li>" << key << ": " << table.tbl.at(key) << "</li>\n";
    }
  }

  html << "</ul>\n</div>\n";
  return html.str();
}

inline std::string generate_pretty(const pretty_data &pd, int index) {
  std::ostringstream html;
  html << "<div class='collapsible'>v Layer " << index << "</div>\n";
  html << "<div class='content'>\n";
  html << generate_table_html("v  CONV", pd.conv);
  html << generate_table_html("v  Tail Block", pd.tailblock);
  html << generate_table_html("v  Output Block", pd.outputblock);
  html << generate_table_html("v  FC", pd.fc);
  html << generate_table_html("v  START", pd.start);
  html << generate_table_html("v  NMS", pd.nms);
  html << generate_table_html("v  Elt Wise", pd.eltwise);
  html << generate_table_html("v  TRANSPOSE", pd.transpose);
  html << generate_table_html("v  RESHAPE", pd.reshape);
  html << "</div>\n";
  return html.str();
}

inline void generate_html(const std::vector<pretty_data> &data,
                          const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    log_fatal("Could not open file {}\n", filename);
  }

  file << "<!DOCTYPE html>\n<html>\n<head>\n";
  file << "<style>\n";
  file << "body { font-family: monospace; }\n";
  file << ".collapsible { cursor: pointer; padding: 10px; background-color: #f1f1f1; border: 1px solid #ddd; margin-top: 5px; }\n";
  file << ".content { display: none; padding: 10px; border-left: 1px solid #ddd; margin-left: 10px; }\n";
  file << ".content ul { list-style-type: none; padding-left: 0; }\n";
  file << "</style>\n";
  file << "<script>\n";
  file << "document.addEventListener('DOMContentLoaded', function() {\n";
  file << "  const coll = document.querySelectorAll('.collapsible');\n";
  file << "  coll.forEach(function(el) {\n";
  file << "    el.addEventListener('click', function() {\n";
  file << "      this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'block' ? 'none' : 'block';\n";
  file << "    });\n";
  file << "  });\n";
  file << "});\n";
  file << "</script>\n";
  file << "</head>\n<body>\n";

  for (size_t i = 0; i < data.size(); i++) {
    file << generate_pretty(data[i], i);
  }

  file << "</body>\n</html>\n";

  file.close();
  std::cout << "HTML file generated: " << filename << "\n";
  std::cout << "Run python -m http.server 5587 to start the server\n";
  std::cout << "Open http://localhost:5587/pretty-print.html in your browser to view it\n";
}
