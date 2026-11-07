#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_gfk_exp.metalh"
#include "hexv842_sw_gt_aod_f32_bindings.metalh"
#include "hexv842_sw_aerosol_f32_bindings.metalh"
#pragma clang fp contract(off)
#define AER3_NRH 8
static inline float sw_exp(float x) {return as_type<float>(soft_gfk_exp(as_type<uint>(x)));}
constant float MPAS_RAOD[3][8][14] = {{{0.0735f,0.0997f,0.1281f,0.1529f,0.1882f,0.2512f,0.3010f,0.4550f,0.7159f,1.0357f,1.3582f,1.6760f,2.2523f,0.0582f},{0.0741f,0.1004f,0.1289f,0.1537f,0.1891f,0.2522f,0.3021f,0.4560f,0.7166f,1.0351f,1.3547f,1.6687f,2.2371f,0.0587f},{0.0752f,0.1017f,0.1304f,0.1554f,0.1909f,0.2542f,0.3042f,0.4580f,0.7179f,1.0342f,1.3485f,1.6559f,2.2102f,0.0596f},{0.0766f,0.1034f,0.1323f,0.1575f,0.1932f,0.2567f,0.3068f,0.4605f,0.7196f,1.0332f,1.3411f,1.6407f,2.1785f,0.0608f},{0.0807f,0.1083f,0.1379f,0.1635f,0.1998f,0.2639f,0.3143f,0.4677f,0.7244f,1.0305f,1.3227f,1.6031f,2.1006f,0.0644f},{0.0884f,0.1174f,0.1482f,0.1746f,0.2118f,0.2769f,0.3277f,0.4805f,0.7328f,1.0272f,1.2977f,1.5525f,1.9976f,0.0712f},{0.1072f,0.1391f,0.1724f,0.2006f,0.2396f,0.3066f,0.3581f,0.5087f,0.7510f,1.0231f,1.2622f,1.4818f,1.8565f,0.0878f},{0.1286f,0.1635f,0.1991f,0.2288f,0.2693f,0.3377f,0.3895f,0.5372f,0.7686f,1.0213f,1.2407f,1.4394f,1.7739f,0.1072f}},{{0.1244f,0.1587f,0.1939f,0.2233f,0.2635f,0.3317f,0.3835f,0.5318f,0.7653f,1.0344f,1.3155f,1.5885f,2.0706f,0.1033f},{0.1159f,0.1491f,0.1834f,0.2122f,0.2518f,0.3195f,0.3712f,0.5207f,0.7585f,1.0331f,1.3130f,1.5833f,2.0601f,0.0956f},{0.1093f,0.1416f,0.1752f,0.2035f,0.2427f,0.3099f,0.3615f,0.5118f,0.7529f,1.0316f,1.3083f,1.5739f,2.0408f,0.0898f},{0.1062f,0.1381f,0.1712f,0.1993f,0.2382f,0.3052f,0.3567f,0.5074f,0.7501f,1.0302f,1.3025f,1.5620f,2.0168f,0.0870f},{0.1045f,0.1361f,0.1690f,0.1970f,0.2357f,0.3025f,0.3540f,0.5049f,0.7486f,1.0271f,1.2864f,1.5297f,1.9518f,0.0854f},{0.1065f,0.1384f,0.1716f,0.1997f,0.2386f,0.3056f,0.3571f,0.5078f,0.7504f,1.0227f,1.2603f,1.4780f,1.8492f,0.0872f},{0.1147f,0.1478f,0.1820f,0.2107f,0.2503f,0.3179f,0.3696f,0.5192f,0.7575f,1.0146f,1.2116f,1.3830f,1.6658f,0.0946f},{0.1247f,0.1590f,0.1943f,0.2237f,0.2639f,0.3322f,0.3840f,0.5322f,0.7656f,1.0082f,1.1719f,1.3075f,1.5252f,0.1036f}},{{0.3053f,0.3507f,0.3932f,0.4261f,0.4681f,0.5334f,0.5797f,0.6962f,0.8583f,1.0187f,1.1705f,1.3049f,1.5205f,0.2748f},{0.3566f,0.4023f,0.4443f,0.4765f,0.5170f,0.5792f,0.6227f,0.7298f,0.8756f,1.0162f,1.1472f,1.2614f,1.4415f,0.3256f},{0.4359f,0.4803f,0.5203f,0.5505f,0.5879f,0.6441f,0.6828f,0.7756f,0.8985f,1.0135f,1.1198f,1.2109f,1.3518f,0.4051f},{0.5128f,0.5544f,0.5913f,0.6187f,0.6523f,0.7020f,0.7358f,0.8149f,0.9174f,1.0115f,1.0995f,1.1740f,1.2875f,0.4835f},{0.6479f,0.6816f,0.7108f,0.7320f,0.7575f,0.7946f,0.8193f,0.8752f,0.9455f,1.0092f,1.0728f,1.1263f,1.2061f,0.6236f},{0.7582f,0.7831f,0.8043f,0.8196f,0.8377f,0.8636f,0.8806f,0.9184f,0.9649f,1.0080f,1.0564f,1.0973f,1.1576f,0.7399f},{0.8482f,0.8647f,0.8785f,0.8884f,0.9000f,0.9164f,0.9272f,0.9506f,0.9789f,1.0072f,1.0454f,1.0780f,1.1256f,0.8360f},{0.8836f,0.8965f,0.9073f,0.9149f,0.9239f,0.9365f,0.9448f,0.9626f,0.9841f,1.0069f,1.0415f,1.0712f,1.1145f,0.8741f}}};
constant float MPAS_SSA[3][8][14] = {{{0.8730f,0.6695f,0.8530f,0.8601f,0.8365f,0.7949f,0.8113f,0.8810f,0.9305f,0.9436f,0.9532f,0.9395f,0.8007f,0.8634f},{0.8428f,0.6395f,0.8571f,0.8645f,0.8408f,0.8007f,0.8167f,0.8845f,0.9326f,0.9454f,0.9545f,0.9416f,0.8070f,0.8589f},{0.8000f,0.6025f,0.8668f,0.8740f,0.8503f,0.8140f,0.8309f,0.8943f,0.9370f,0.9489f,0.9577f,0.9451f,0.8146f,0.8548f},{0.7298f,0.5666f,0.9030f,0.9049f,0.8863f,0.8591f,0.8701f,0.9178f,0.9524f,0.9612f,0.9677f,0.9576f,0.8476f,0.8578f},{0.7010f,0.5606f,0.9312f,0.9288f,0.9183f,0.9031f,0.9112f,0.9439f,0.9677f,0.9733f,0.9772f,0.9699f,0.8829f,0.8590f},{0.6933f,0.5620f,0.9465f,0.9393f,0.9346f,0.9290f,0.9332f,0.9549f,0.9738f,0.9782f,0.9813f,0.9750f,0.8980f,0.8594f},{0.6842f,0.5843f,0.9597f,0.9488f,0.9462f,0.9470f,0.9518f,0.9679f,0.9808f,0.9839f,0.9864f,0.9794f,0.9113f,0.8648f},{0.6786f,0.5897f,0.9658f,0.9522f,0.9530f,0.9610f,0.9651f,0.9757f,0.9852f,0.9871f,0.9883f,0.9835f,0.9236f,0.8618f}},{{0.4063f,0.3663f,0.4093f,0.4205f,0.4487f,0.4912f,0.5184f,0.5743f,0.6233f,0.6392f,0.6442f,0.6408f,0.6105f,0.4094f},{0.4113f,0.3654f,0.4215f,0.4330f,0.4604f,0.5022f,0.5293f,0.5848f,0.6336f,0.6493f,0.6542f,0.6507f,0.6205f,0.4196f},{0.4500f,0.3781f,0.4924f,0.5050f,0.5265f,0.5713f,0.6048f,0.6274f,0.6912f,0.7714f,0.7308f,0.7027f,0.6772f,0.4820f},{0.5075f,0.4139f,0.5994f,0.6127f,0.6350f,0.6669f,0.6888f,0.7333f,0.7704f,0.7809f,0.7821f,0.7762f,0.7454f,0.5709f},{0.5596f,0.4570f,0.7009f,0.7118f,0.7317f,0.7583f,0.7757f,0.8093f,0.8361f,0.8422f,0.8406f,0.8337f,0.8036f,0.6525f},{0.6008f,0.4971f,0.7845f,0.7906f,0.8075f,0.8290f,0.8418f,0.8649f,0.8824f,0.8849f,0.8815f,0.8739f,0.8455f,0.7179f},{0.6401f,0.5407f,0.8681f,0.8664f,0.8796f,0.8968f,0.9043f,0.9159f,0.9244f,0.9234f,0.9182f,0.9105f,0.8849f,0.7796f},{0.6567f,0.5618f,0.9073f,0.9077f,0.9182f,0.9279f,0.9325f,0.9398f,0.9440f,0.9413f,0.9355f,0.9278f,0.9039f,0.8040f}},{{0.9697f,0.9183f,0.9749f,0.9820f,0.9780f,0.9712f,0.9708f,0.9778f,0.9831f,0.9827f,0.9826f,0.9723f,0.8763f,0.9716f},{0.9070f,0.8491f,0.9730f,0.9816f,0.9804f,0.9742f,0.9738f,0.9802f,0.9847f,0.9841f,0.9838f,0.9744f,0.8836f,0.9546f},{0.8378f,0.7761f,0.9797f,0.9827f,0.9829f,0.9814f,0.9812f,0.9852f,0.9882f,0.9875f,0.9871f,0.9791f,0.9006f,0.9348f},{0.7866f,0.7249f,0.9890f,0.9822f,0.9856f,0.9917f,0.9924f,0.9932f,0.9943f,0.9938f,0.9933f,0.9887f,0.9393f,0.9204f},{0.7761f,0.7164f,0.9959f,0.9822f,0.9834f,0.9941f,0.9955f,0.9952f,0.9960f,0.9956f,0.9951f,0.9922f,0.9538f,0.9152f},{0.7671f,0.7114f,0.9902f,0.9786f,0.9838f,0.9954f,0.9970f,0.9965f,0.9971f,0.9968f,0.9964f,0.9943f,0.9644f,0.9158f},{0.7551f,0.7060f,0.9890f,0.9743f,0.9807f,0.9966f,0.9989f,0.9978f,0.9982f,0.9980f,0.9978f,0.9964f,0.9757f,0.9122f},{0.7439f,0.7000f,0.9870f,0.9695f,0.9769f,0.9970f,1.0000f,0.9984f,0.9988f,0.9986f,0.9984f,0.9975f,0.9825f,0.9076f}}};
constant float MPAS_ASY[3][8][14] = {{{0.7444f,0.7711f,0.7306f,0.7103f,0.6693f,0.6267f,0.6169f,0.6207f,0.6341f,0.6497f,0.6630f,0.6748f,0.7208f,0.7419f},{0.7444f,0.7747f,0.7314f,0.7110f,0.6711f,0.6301f,0.6210f,0.6251f,0.6392f,0.6551f,0.6680f,0.6799f,0.7244f,0.7436f},{0.7438f,0.7845f,0.7341f,0.7137f,0.6760f,0.6381f,0.6298f,0.6350f,0.6497f,0.6657f,0.6790f,0.6896f,0.7300f,0.7477f},{0.7336f,0.7934f,0.7425f,0.7217f,0.6925f,0.6665f,0.6616f,0.6693f,0.6857f,0.7016f,0.7139f,0.7218f,0.7495f,0.7574f},{0.7111f,0.7865f,0.7384f,0.7198f,0.6995f,0.6864f,0.6864f,0.6987f,0.7176f,0.7326f,0.7427f,0.7489f,0.7644f,0.7547f},{0.7009f,0.7828f,0.7366f,0.7196f,0.7034f,0.6958f,0.6979f,0.7118f,0.7310f,0.7452f,0.7542f,0.7593f,0.7692f,0.7522f},{0.7226f,0.8127f,0.7621f,0.7434f,0.7271f,0.7231f,0.7248f,0.7351f,0.7506f,0.7622f,0.7688f,0.7719f,0.7756f,0.7706f},{0.7296f,0.8219f,0.7651f,0.7513f,0.7404f,0.7369f,0.7386f,0.7485f,0.7626f,0.7724f,0.7771f,0.7789f,0.7790f,0.7760f}},{{0.7399f,0.7372f,0.7110f,0.6916f,0.6582f,0.6230f,0.6147f,0.6214f,0.6412f,0.6655f,0.6910f,0.7124f,0.7538f,0.7395f},{0.7400f,0.7419f,0.7146f,0.6952f,0.6626f,0.6287f,0.6209f,0.6280f,0.6481f,0.6723f,0.6974f,0.7180f,0.7575f,0.7432f},{0.7363f,0.7614f,0.7303f,0.7100f,0.6815f,0.6550f,0.6498f,0.6590f,0.6802f,0.7032f,0.7255f,0.7430f,0.7735f,0.7580f},{0.7180f,0.7701f,0.7358f,0.7163f,0.6952f,0.6807f,0.6801f,0.6935f,0.7160f,0.7370f,0.7553f,0.7681f,0.7862f,0.7623f},{0.7013f,0.7733f,0.7374f,0.7203f,0.7057f,0.7006f,0.7035f,0.7192f,0.7415f,0.7596f,0.7739f,0.7827f,0.7906f,0.7596f},{0.6922f,0.7773f,0.7404f,0.7264f,0.7170f,0.7179f,0.7228f,0.7389f,0.7595f,0.7746f,0.7851f,0.7909f,0.7918f,0.7562f},{0.6928f,0.7875f,0.7491f,0.7393f,0.7345f,0.7397f,0.7455f,0.7602f,0.7773f,0.7883f,0.7944f,0.7970f,0.7912f,0.7555f},{0.7021f,0.7989f,0.7590f,0.7512f,0.7613f,0.7746f,0.7718f,0.7727f,0.7867f,0.7953f,0.7988f,0.7994f,0.7906f,0.7600f}},{{0.6620f,0.7011f,0.7111f,0.7068f,0.6990f,0.6918f,0.6883f,0.6827f,0.6768f,0.6773f,0.6863f,0.6940f,0.7245f,0.6719f},{0.6880f,0.7394f,0.7297f,0.7240f,0.7162f,0.7083f,0.7038f,0.6957f,0.6908f,0.6917f,0.6952f,0.7035f,0.7356f,0.6977f},{0.7266f,0.7970f,0.7666f,0.7593f,0.7505f,0.7427f,0.7391f,0.7293f,0.7214f,0.7210f,0.7212f,0.7265f,0.7519f,0.7340f},{0.7683f,0.8608f,0.8120f,0.8030f,0.7826f,0.7679f,0.7713f,0.7760f,0.7723f,0.7716f,0.7726f,0.7767f,0.7884f,0.7768f},{0.7776f,0.8727f,0.8182f,0.8083f,0.7985f,0.7939f,0.7953f,0.7913f,0.7846f,0.7870f,0.7899f,0.7918f,0.7969f,0.7870f},{0.7878f,0.8839f,0.8231f,0.8130f,0.8050f,0.7977f,0.7945f,0.7932f,0.7955f,0.7992f,0.8025f,0.8035f,0.8055f,0.7956f},{0.8005f,0.8957f,0.8273f,0.8179f,0.8105f,0.8035f,0.8010f,0.8030f,0.8081f,0.8108f,0.8143f,0.8174f,0.8174f,0.8042f},{0.8104f,0.9034f,0.8294f,0.8212f,0.8144f,0.8087f,0.8077f,0.8118f,0.8175f,0.8202f,0.8239f,0.8265f,0.8246f,0.8095f}}};
constant float MPAS_RHS[8]={0.f,50.f,70.f,80.f,90.f,95.f,98.f,99.f};
constant float AER3_GT_RH_ARR[AER3_NRH] = {
    10.f, 60.f, 70.f, 80.f, 85.f, 90.f, 95.f, 99.8f};
constant float AER3_GT_LOOKUP[AER3_NRH][4][2] = {
    {{5.73936E-15f, 2.63577E-12f}, {5.73936E-15f, 2.63577E-12f},
     {5.73936E-15f, 2.63577E-12f}, {5.73936E-15f, 2.63577E-12f}},
    {{6.93515E-15f, 2.72095E-12f}, {6.93168E-15f, 2.72092E-12f},
     {6.92570E-15f, 2.72091E-12f}, {6.91833E-15f, 2.72087E-12f}},
    {{7.24707E-15f, 2.77219E-12f}, {7.23809E-15f, 2.77222E-12f},
     {7.23108E-15f, 2.77201E-12f}, {7.21800E-15f, 2.77111E-12f}},
    {{8.95130E-15f, 2.87263E-12f}, {9.01582E-15f, 2.87252E-12f},
     {9.13216E-15f, 2.87241E-12f}, {9.16219E-15f, 2.87211E-12f}},
    {{1.06695E-14f, 2.96752E-12f}, {1.06370E-14f, 2.96726E-12f},
     {1.05999E-14f, 2.96702E-12f}, {1.05443E-14f, 2.96603E-12f}},
    {{1.37908E-14f, 3.15081E-12f}, {1.37172E-14f, 3.15020E-12f},
     {1.36362E-14f, 3.14927E-12f}, {1.35287E-14f, 3.14817E-12f}},
    {{2.26019E-14f, 3.66798E-12f}, {2.24435E-14f, 3.66540E-12f},
     {2.23254E-14f, 3.66173E-12f}, {2.20496E-14f, 3.65796E-12f}},
    {{4.41983E-13f, 7.50091E-11f}, {3.93335E-13f, 6.79097E-11f},
     {3.45569E-13f, 6.07845E-11f}, {2.96971E-13f, 5.36085E-11f}}};
static inline float aer3_rslf(float p, float t)
{
    const float c0 = .611583699E03f, c1 = .444606896E02f,
                c2 = .143177157E01f, c3 = .264224321E-1f,
                c4 = .299291081E-3f, c5 = .203154182E-5f,
                c6 = .702620698E-8f, c7 = .379534310E-11f,
                c8 = -.321582393E-13f;
    float x = mpas_max(-80.0f, mpas_sub(t, 273.16f));
    float esl = mpas_add(c7, mpas_mul(x, c8));
    esl = mpas_add(c6, mpas_mul(x, esl));
    esl = mpas_add(c5, mpas_mul(x, esl));
    esl = mpas_add(c4, mpas_mul(x, esl));
    esl = mpas_add(c3, mpas_mul(x, esl));
    esl = mpas_add(c2, mpas_mul(x, esl));
    esl = mpas_add(c1, mpas_mul(x, esl));
    esl = mpas_add(c0, mpas_mul(x, esl));
    esl = mpas_min(esl, mpas_mul(p, 0.15f));
    return mpas_div(mpas_mul(0.622f, esl), mpas_sub(p, esl));
}

// MPAS module_mp_thompson_aerosols.F:134-162, integer NINT indexing.
static inline int aer3_nint(float x)
{
    return (int)round(x);
}

// MPAS module_mp_thompson_aerosols.F:23-210, gt_aod for one point.
static inline float aer3_gt_aod(float p, float dz8w, float t, float qv,
                                    float nwfa, float nifa)
{
    const int rind = AER3_NRH;
    float rhoa = mpas_div(p, mpas_mul(287.0f, t));
    int t_idx = max(1, min(aer3_nint(mpas_sub(10.999f, mpas_mul(0.0333f, t))), 4));
    float qvsat = aer3_rslf(p, t);
    float rh = mpas_min(98.0f, mpas_max(10.1f, mpas_mul(mpas_div(qv, qvsat), 100.0f)));
    int i1, i2;
    if (rh < 60.0f) {
        i1 = 1;
        i2 = 2;
    } else {
        int idx;
        if (rh >= 60.0f && rh < 80.0f) {
            idx = aer3_nint(mpas_add(mpas_mul(0.1f, rh), -4.0f));
        } else {
            idx = min(rind, aer3_nint(mpas_add(mpas_mul(0.2f, rh), -12.0f)));
        }
        float rh_d = mpas_sub(rh, AER3_GT_RH_ARR[idx - 1]);
        if (rh_d < 0.0f) {
            i1 = idx - 1;
            i2 = idx;
        } else {
            i1 = idx;
            i2 = idx + 1;
            if (i2 > rind) {
                i2 = rind;
                i1 = rind - 1;
            }
        }
    }
    float a1 = AER3_GT_RH_ARR[i1 - 1];
    float a2 = AER3_GT_RH_ARR[i2 - 1];
    float rh_f = mpas_max(0.0f, mpas_min(1.0f, mpas_div(
        mpas_sub(mpas_div(rh, mpas_sub(100.0f, rh)), mpas_div(a1, mpas_sub(100.0f, a1))),
        mpas_sub(mpas_div(a2, mpas_sub(100.0f, a2)), mpas_div(a1, mpas_sub(100.0f, a1))))));
    float l1w = AER3_GT_LOOKUP[i1 - 1][t_idx - 1][0];
    float l2w = AER3_GT_LOOKUP[i2 - 1][t_idx - 1][0];
    float l1i = AER3_GT_LOOKUP[i1 - 1][t_idx - 1][1];
    float l2i = AER3_GT_LOOKUP[i2 - 1][t_idx - 1][1];
    float unit_bext1 = mpas_add(l1w, mpas_mul(mpas_sub(l2w, l1w), rh_f));
    float unit_bext3 = mpas_add(l1i, mpas_mul(mpas_sub(l2i, l1i), rh_f));
    float ntemp = mpas_max(1.0f, mpas_min(99999.E6f, nwfa));
    float aod_wfa = mpas_mul(mpas_mul(mpas_mul(unit_bext1, ntemp), dz8w), rhoa);
    ntemp = mpas_max(0.01f, mpas_min(9999.E6f, nifa));
    float aod_ifa = mpas_mul(mpas_mul(mpas_mul(unit_bext3, ntemp), dz8w), rhoa);
    return mpas_add(aod_wfa, aod_ifa);
}



static inline float mpas_sw_rh(float p,float t,float qv) {
 float tc=mpas_sub(t,273.15f),rv=mpas_max(0.f,qv);
 float es=mpas_mul(6.112f,sw_exp(mpas_div(mpas_mul(17.6f,tc),mpas_add(tc,243.5f))));
 float e=mpas_div(mpas_mul(mpas_mul(.01f,rv),p),mpas_add(rv,.62197f));
 return mpas_min(99.f,mpas_max(0.f,mpas_div(mpas_mul(100.f,e),es)));
}
// MPAS module_ra_rrtmg_sw_aerosols.F:453-607,609-722,724-837.
static inline float mpas_sw_spectral(constant float table[3][8][14],int type,int band,float rh,float aod,bool multiply) {
 int ii=0;while(ii<8&&rh>MPAS_RHS[ii])ii++;
 int imin=max(0,ii-3),imax=min(7,ii+2);float total=0.f;
 for(int jj=imin;jj<=imax;jj++){
  float lj=1.f;for(int k=imin;k<=imax;k++)if(k!=jj)lj=mpas_div(mpas_mul(lj,mpas_sub(rh,MPAS_RHS[k])),mpas_sub(MPAS_RHS[jj],MPAS_RHS[k]));
  float term=mpas_mul(lj,table[type-1][jj][band]);if(multiply)term=mpas_mul(term,aod);total=mpas_add(total,term);
 }
 return total;
}

kernel void hexv842_sw_gt_aod_f32(constant Hexv842SwGtAodF32Resources &r [[buffer(0)]],
 constant Hexv842SwGtAodF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int c=int(tid),nz=p.nz;if(c>=p.nc)return;
 if(r.xland[c]==1.f)r.type[c]=1;else if(r.xland[c]==2.f)r.type[c]=3;else {r.invalid[c]=1;return;}
 float sum=0.f;for(int k=0;k<nz;k++){int i=c*nz+k;r.aod3[i]=aer3_gt_aod(r.pressure[i],r.dz[i],r.temperature[i],r.qv[i],r.nwfa[i],r.nifa[i]);sum=mpas_add(sum,r.aod3[i]);}r.aod2[c]=sum;
}
kernel void hexv842_sw_aerosol_f32(constant Hexv842SwAerosolF32Resources &r [[buffer(0)]],
 constant Hexv842SwAerosolF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int c=int(tid),nz=p.nz,nl=p.nl;if(c>=p.nc)return;
 for(int k=0;k<nl;k++){
  if(k>=nz){for(int b=0;b<14;b++){int i=(c*14+b)*nl+k;r.tau[i]=0.f;r.ssa[i]=1.f;r.asy[i]=0.f;}continue;}
  float rh=mpas_sw_rh(r.pressure[c*nz+k],r.temperature[c*nz+k],r.qv[c*nz+k]);
  for(int b=0;b<14;b++){
   int i=(c*14+b)*nl+k;
   r.tau[i]=mpas_sw_spectral(MPAS_RAOD,r.type[c],b,rh,r.aod3[c*nz+k],true);
   r.ssa[i]=mpas_sw_spectral(MPAS_SSA,r.type[c],b,rh,0.f,false);
   r.asy[i]=mpas_sw_spectral(MPAS_ASY,r.type[c],b,rh,0.f,false);
  }
 }
}
