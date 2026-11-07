#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "thompson_column_helpers.metalh"
using namespace ntc;
using tc44::cd;
using tc44::refl_rayleigh_soak_wetgraupel;
#define TCKMAX nz
#define TCSPACE device
#define TCF(n,sz) device F * n=tc_wsf;tc_wsf+=(sz)
#define TCD(n,sz) device Soft8 * n=tc_wsd;tc_wsd+=(sz)
#define TCB(n,sz) device bool * n=tc_wsb;tc_wsb+=(sz)
#define TCKF(n,sz) device F * n=m.tc_wsf;m.tc_wsf+=(sz)
static inline bool tc_any(device bool* a,int lo,int hi){for(int k=lo;k<=hi;k++)if(a[k-lo])return true;return false;}
constant Soft8 tc_xxds [ 50 ] = {
SD8(0x3f1ba409db297486ul) , SD8(0x3f1ebb079752770dul) , SD8(0x3f2115387d5665e6ul) , SD8(0x3f22fe142ad6d844ul) , SD8(0x3f251d95692c1748ul) , SD8(0x3f2779d809403699ul) , SD8(0x3f2a19a6ab6f8cf9ul) , SD8(0x3f2d048e4a0ce200ul) , SD8(0x3f302179f98cdd44ul) , SD8(0x3f31ef1677dab285ul) , SD8(0x3f33f04ccf611fb4ul) , SD8(0x3f362ae1a802bfa6ul) , SD8(0x3f38a53ebad47393ul) , SD8(0x3f3b668545d00f11ul) , SD8(0x3f3e76a28f903d41ul) , SD8(0x3f40ef335b0e2af3ul) , SD8(0x3f42d3cf0531fea0ul) , SD8(0x3f44ee96a0b0304cul) , SD8(0x3f47459865fc6260ul) , SD8(0x3f49df8fd7ecb26eul) , SD8(0x3f4cc3f922c031beul) , SD8(0x3f4ffb26a57ce929ul) , SD8(0x3f51c72c71ce74e4ul) , SD8(0x3f53c3ec91f79b6bul) , SD8(0x3f55f98b84a26d3dul) , SD8(0x3f586e64bd092f83ul) , SD8(0x3f5b29899ad9698aul) , SD8(0x3f5e32d5c0476ac2ul) , SD8(0x3f60c982d70b9635ul) , SD8(0x3f62a9e7f3542e0cul) , SD8(0x3f64c000702c7f67ul) , SD8(0x3f6711cd0bd3ecb9ul) , SD8(0x3f69a5fa4d3e7318ul) , SD8(0x3f6c83f3b7fe38fful) , SD8(0x3f6fb3f925b2e709ul) , SD8(0x3f719f9b413630ebul) , SD8(0x3f7397ef18262e6ful) , SD8(0x3f75c8a32f2a4f82ul) , SD8(0x3f783804d36655c9ul) , SD8(0x3f7aed15a98c9f98ul) , SD8(0x3f7def9fd6aeec8dul) , SD8(0x3f80a42634faa2eful) , SD8(0x3f82805e23dc0717ul) , SD8(0x3f8491d1eed7d837ul) , SD8(0x3f86de74f7f81201ul) , SD8(0x3f896ce4eba7ca80ul) , SD8(0x3f8c447cc9dfcb0ful) , SD8(0x3f8f6d6a1011ebcaul) , SD8(0x3f917862205c08b5ul) , SD8(0x3f936c53861d22d4ul) }
;

constant Soft8 tc_xxdg [ 50 ] = {
SD8(0x3f1be52c6d4d4e0bul) , SD8(0x3f1f9648b0c9dd5dul) , SD8(0x3f21e23cff932616ul) , SD8(0x3f2440271c8b315bul) , SD8(0x3f26ee420edffc82ul) , SD8(0x3f29f72ac2b16e9ful) , SD8(0x3f2d66e5b77868e7ul) , SD8(0x3f30a5874b5b6693ul) , SD8(0x3f32d9870b7f02abul) , SD8(0x3f35582b8057a06bul) , SD8(0x3f382b55a30b2b62ul) , SD8(0x3f3b5e35203d655ful) , SD8(0x3f3efd74a3fc3bd5ul) , SD8(0x3f418bb6013c913bul) , SD8(0x3f43de2c8217d90bul) , SD8(0x3f467f4fdb31b216ul) , SD8(0x3f4979899f513f38ul) , SD8(0x3f4cd8a428d880e8ul) , SD8(0x3f5054fca51a4b10ul) , SD8(0x3f527e539615bdb2ul) , SD8(0x3f54f0e615b10661ul) , SD8(0x3f57b6655115131dul) , SD8(0x3f5ad9cad5844970ul) , SD8(0x3f5e678405f4a72aul) , SD8(0x3f6136d1a88cc628ul) , SD8(0x3f637e0bf55e34bbul) , SD8(0x3f6612767286fb31ul) , SD8(0x3f68fe4851e6e59cul) , SD8(0x3f6c4d12e21bf0e1ul) , SD8(0x3f7005f7ae866566ul) , SD8(0x3f7224d963359ebdul) , SD8(0x3f748b9453ca2efeul) , SD8(0x3f7743aac8b1aefcul) , SD8(0x3f7a57e135a9f49cul) , SD8(0x3f7dd468dd15f31bul) , SD8(0x3f80e3880bf712d4ul) , SD8(0x3f831fbc80be7278ul) , SD8(0x3f85a7abafb253feul) , SD8(0x3f88855b5d8a8dfbul) , SD8(0x3f8bc424e122d453ul) , SD8(0x3f8f70e214646e8eul) , SD8(0x3f91cd101be97692ul) , SD8(0x3f94282cc91ff6b1ul) , SD8(0x3f96d31b586abf0aul) , SD8(0x3f99d86c24ece8bbul) , SD8(0x3f9d4415736099daul) , SD8(0x3fa091d16837085eul) , SD8(0x3fa2c33559f20ba0ul) , SD8(0x3fa53ee59e9c3495ul) , SD8(0x3fa80eb77ce9713ful) }
;

constant Soft8 tc_xdts [ 50 ] = {
SD8(0x3ee77161f03b8d68ul) , SD8(0x3eea103e6f958e30ul) , SD8(0x3eecfa18da335f10ul) , SD8(0x3ef01ba99bc1ebd0ul) , SD8(0x3ef1e89fb9bee440ul) , SD8(0x3ef3e91d17cbba40ul) , SD8(0x3ef622e44991cba8ul) , SD8(0x3ef89c5cb86e2e28ul) , SD8(0x3efb5ca5107dd508ul) , SD8(0x3efe6ba7bcf49940ul) , SD8(0x3f00e918d7584e60ul) , SD8(0x3f02cd05d7491c10ul) , SD8(0x3f04e70b4239fe30ul) , SD8(0x3f073d3521e19ef8ul) , SD8(0x3f09d63c8be3d3e0ul) , SD8(0x3f0cb99af9dcbc90ul) , SD8(0x3f0fef9fcafb9c00ul) , SD8(0x3f11c0c416875130ul) , SD8(0x3f13bcccd8e5ca20ul) , SD8(0x3f15f19fee6907a0ul) , SD8(0x3f1865967fb87488ul) , SD8(0x3f1b1fbf605cdee8ul) , SD8(0x3f1e27f35d7f9700ul) , SD8(0x3f20c375e8e694d8ul) , SD8(0x3f22a32ddfb908e0ul) , SD8(0x3f24b885dc355db0ul) , SD8(0x3f27097c72b987d8ul) , SD8(0x3f299cbbc26dc8e0ul) , SD8(0x3f2c79aca2426cc0ul) , SD8(0x3f2fa88bf2ad43b0ul) , SD8(0x3f31994128bf2170ul) , SD8(0x3f3390df39ff1ab0ul) , SD8(0x3f35c0c93994f538ul) , SD8(0x3f382f4a2f2b1680ul) , SD8(0x3f3ae36138fa9020ul) , SD8(0x3f3de4d5ad57ce60ul) , SD8(0x3f409e26be2abe10ul) , SD8(0x3f4279b308f1b160ul) , SD8(0x3f448a680000f378ul) , SD8(0x3f46d636e051cc08ul) , SD8(0x3f4963baf3de1ce0ul) , SD8(0x3f4c3a4c93f09170ul) , SD8(0x3f4f62164b6f3720ul) , SD8(0x3f5172162af7b9a8ul) , SD8(0x3f5365535f972f98ul) , SD8(0x3f55905f370ba5e0ul) , SD8(0x3f57f976b77487e8ul) , SD8(0x3f5aa7896cb11678ul) , SD8(0x3f5da24d5d1f3a40ul) , SD8(0x3f60792a9cb73498ul) }
;

constant Soft8 tc_xdtg [ 50 ] = {
SD8(0x3eebc14e866a2140ul) , SD8(0x3eef6dab983eb268ul) , SD8(0x3ef1cb3e70a34428ul) , SD8(0x3ef4261d7ca34128ul) , SD8(0x3ef6d0c642a7d4a8ul) , SD8(0x3ef9d5c8097b0790ul) , SD8(0x3efd4117dceb0db0ul) , SD8(0x3f00901ff3a518e8ul) , SD8(0x3f02c14a87960f38ul) , SD8(0x3f053cb9d6d8b830ul) , SD8(0x3f080c4226e5e400ul) , SD8(0x3f0b3b04c25b0230ul) , SD8(0x3f0ed59c0bf1f4e0ul) , SD8(0x3f117526b35e1998ul) , SD8(0x3f13c4a0dca10fd8ul) , SD8(0x3f166262b5a79fb0ul) , SD8(0x3f1958c86de8ede0ul) , SD8(0x3f1cb38d36df1978ul) , SD8(0x3f203ffcdc37b48cul) , SD8(0x3f22668c559f03b8ul) , SD8(0x3f24d5f9349d7a20ul) , SD8(0x3f2797e8302391e8ul) , SD8(0x3f2ab744b9043188ul) , SD8(0x3f2e406c37adda40ul) , SD8(0x3f3120af81772098ul) , SD8(0x3f3364fbe8d28898ul) , SD8(0x3f35f6154179d6a8ul) , SD8(0x3f38de259ac34ad0ul) , SD8(0x3f3c28af63be1408ul) , SD8(0x3f3fe2bafed72e80ul) , SD8(0x3f420d852ed55ae8ul) , SD8(0x3f447129b8afb4a0ul) , SD8(0x3f4725c12b7b19a8ul) , SD8(0x3f4a360222cc0320ul) , SD8(0x3f4dae0e33cd3220ul) , SD8(0x3f50cdd0fb8e0b38ul) , SD8(0x3f530725b71ceb38ul) , SD8(0x3f558bd3cdfc5510ul) , SD8(0x3f5865d421e79ea8ul) , SD8(0x3f5ba0717221de28ul) , SD8(0x3f5f48751296f8f0ul) , SD8(0x3f61b62cc6fcf5b8ul) , SD8(0x3f640e41fdb0a0c0ul) , SD8(0x3f66b5c27535fa78ul) , SD8(0x3f69b730f38289c8ul) , SD8(0x3f6d1e745bf18448ul) , SD8(0x3f707c8368599540ul) , SD8(0x3f72ab15888706a8ul) , SD8(0x3f75239473e3d5a8ul) , SD8(0x3f77efc8cc80fdb0ul) }
;

constant float tc_ta_ww [ 9 ] = {
as_type<float>(1008981770u) , as_type<float>(1023504128u) , as_type<float>(1036831949u) , as_type<float>(1050790593u) , as_type<float>(1065353216u) , as_type<float>(1078607217u) , as_type<float>(1092616192u) , as_type<float>(1107086541u) , as_type<float>(1120403456u) }
;

constant float tc_ta_tk [ 7 ] = {
as_type<float>(1131619942u) , as_type<float>(1132275302u) , as_type<float>(1132696371u) , as_type<float>(1133024051u) , as_type<float>(1133351731u) , as_type<float>(1133679411u) , as_type<float>(1134007091u) }
;

constant float tc_ta_na [ 7 ] = {
as_type<float>(1092616192u) , as_type<float>(1107086541u) , as_type<float>(1120403456u) , as_type<float>(1134428160u) , as_type<float>(1148846080u) , as_type<float>(1162182656u) , as_type<float>(1176256512u) }
;

constant Soft8 tc_t_nc [ 100 ] = {
SD8(0x412fc397d2e29693ul) , SD8(0x413134b48c43cf34ul) , SD8(0x4132a3dd352f6082ul) , SD8(0x413431a0b96efa44ul) , SD8(0x4135e08c2e2e9a8ful) , SD8(0x4137b36318d4d12aul) , SD8(0x4139ad23f8ae0103ul) , SD8(0x413bd10d316ca88dul) , SD8(0x413e22a25e91018ful) , SD8(0x414052d90c3b435ful) , SD8(0x4141af2e1a4042f1ul) , SD8(0x4143288c45548635ul) , SD8(0x4144c15f25c83e36ul) , SD8(0x41467c45f98cacedul) , SD8(0x41485c17f24e7439ul) , SD8(0x414a63e8df6dd417ul) , SD8(0x414c970e3b7d34d8ul) , SD8(0x414ef924a591c26cul) , SD8(0x4150c70ae7b172b6ul) , SD8(0x41522d0f74fa97d3ul) , SD8(0x4153b0ebcdae8ba0ul) , SD8(0x4155551cc48379beul) , SD8(0x41571c544172c402ul) , SD8(0x4159097dae784d1ful) , SD8(0x415b1fc2c2bd9fe4ul) , SD8(0x415d6290b40fd1e8ul) , SD8(0x415fd59dd726f44dul) , SD8(0x41613e77dbfdec53ul) , SD8(0x4162ae70da17073bul) , SD8(0x41643d161128c52bul) , SD8(0x4165ecf608f160a4ul) , SD8(0x4167c0d5d84d527cul) , SD8(0x4169bbb5b175ba7bul) , SD8(0x416be0d5cf4ac06bul) , SD8(0x416e33bbcbc0d301ul) , SD8(0x41705c1c341a246bul) , SD8(0x4171b936e890f542ul) , SD8(0x4173336b33c6c333ul) , SD8(0x4174cd260d9d958bul) , SD8(0x4176890830e6b719ul) , SD8(0x417869ea6bee5fa3ul) , SD8(0x417a72e24d196d2aul) , SD8(0x417ca7473341e1f9ul) , SD8(0x417f0ab7ca22a1b9ul) , SD8(0x4180d08ffdea29e7ul) , SD8(0x4182375fb08ec365ul) , SD8(0x4183bc181d943659ul) , SD8(0x4185613781090f5ful) , SD8(0x418729714a5dddf0ul) , SD8(0x418917b28ba73cf6ul) , SD8(0x418b2f26c7814b82ul) , SD8(0x418d733d2577ebc4ul) , SD8(0x418fe7ae157f7177ul) , SD8(0x41914840b5dfe1e0ul) , SD8(0x4192b90a7f5cc246ul) , SD8(0x41944891e9516a9dul) , SD8(0x4195f966eee0969ful) , SD8(0x4197ce503940740cul) , SD8(0x4199ca4fae8da851ul) , SD8(0x419bf0a761e15f8ful) , SD8(0x419e44deecc5e17ful) , SD8(0x41a065649d6914b2ul) , SD8(0x41a1c345687835adul) , SD8(0x41a33e504d4f3eb1ul) , SD8(0x41a4d8f3a4298a05ul) , SD8(0x41a695d1a59088e2ul) , SD8(0x41a877c4bd5a2a50ul) , SD8(0x41aa81e439ed8fc5ul) , SD8(0x41acb7895f7f1f8aul) , SD8(0x41af1c54e79926e0ul) , SD8(0x41b0da1a7afc0728ul) , SD8(0x41b241b5c640e2d5ul) , SD8(0x41b3c74ac4785f82ul) , SD8(0x41b56d591bd6736ful) , SD8(0x41b73695c420fbb3ul) , SD8(0x41b925ef7875ef59ul) , SD8(0x41bb3e9387e8b4a4ul) , SD8(0x41bd83f30cde6866ul) , SD8(0x41bff9c893b999e2ul) , SD8(0x41c1520f1d0e5ff1ul) , SD8(0x41c2c3aa28685515ul) , SD8(0x41c45414459958aful) , SD8(0x41c605dee3fb63eeul) , SD8(0x41c7dbd24002a692ul) , SD8(0x41c9d8f1f4a6a06eul) , SD8(0x41cc0081ee4574a8ul) , SD8(0x41ce560bc7218c78ul) , SD8(0x41d06eb24b238155ul) , SD8(0x41d1cd599d311433ul) , SD8(0x41d3493b956df97eul) , SD8(0x41d4e4c7ed36cc62ul) , SD8(0x41d6a2a25ba5bcc4ul) , SD8(0x41d885a6eb05170aul) , SD8(0x41da90eeaabc757ful) , SD8(0x41dcc7d4c56e087bul) , SD8(0x41df2dfc039de18cul) , SD8(0x41e0e3aa61f7b1d9ul) , SD8(0x41e24c11b9630558ul) , SD8(0x41e3d283c5f3f1d1ul) , SD8(0x41e5798198d15441ul) }
;

constant Soft8 tc_simpson [ 51 ] = {
SD8(0x3fd5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fe5555555555555ul) , SD8(0x3ff5555555555555ul) , SD8(0x3fd5555555555555ul) }
;

constant float tc_sb [ 10 ] = {
as_type<float>(1056166717u) , - as_type<float>(1015167060u) , as_type<float>(1042937312u) , as_type<float>(1005893152u) , - as_type<float>(957602150u) , as_type<float>(1031225943u) , as_type<float>(950381773u) , as_type<float>(974894762u) , as_type<float>(0u) , - as_type<float>(996830234u) }
;

constant float tc_sa [ 10 ] = {
as_type<float>(1084364610u) , - as_type<float>(1031820125u) , - as_type<float>(1078071864u) , as_type<float>(1022454009u) , - as_type<float>(966093837u) , as_type<float>(1050674830u) , as_type<float>(961931477u) , as_type<float>(995206736u) , as_type<float>(0u) , - as_type<float>(1015197125u) }
;

constant float tc_r_s [ 28 ] = {
as_type<float>(925353388u) , as_type<float>(933741996u) , as_type<float>(939239554u) , as_type<float>(942130604u) , as_type<float>(944879383u) , as_type<float>(947628162u) , as_type<float>(949144823u) , as_type<float>(950519212u) , as_type<float>(951893602u) , as_type<float>(953267991u) , as_type<float>(961656599u) , as_type<float>(966609234u) , as_type<float>(970045207u) , as_type<float>(973279855u) , as_type<float>(974997842u) , as_type<float>(976715828u) , as_type<float>(978433815u) , as_type<float>(980151802u) , as_type<float>(981668463u) , as_type<float>(990057071u) , as_type<float>(994352038u) , as_type<float>(998445679u) , as_type<float>(1000593162u) , as_type<float>(1002740646u) , as_type<float>(1004888130u) , as_type<float>(1006834287u) , as_type<float>(1007908028u) , as_type<float>(1008981770u) }
;

constant float tc_r_r [ 37 ] = {
as_type<float>(897988541u) , as_type<float>(906377149u) , as_type<float>(910775196u) , as_type<float>(914765757u) , as_type<float>(916964780u) , as_type<float>(919163804u) , as_type<float>(921362827u) , as_type<float>(923154365u) , as_type<float>(924253877u) , as_type<float>(925353388u) , as_type<float>(933741996u) , as_type<float>(939239554u) , as_type<float>(942130604u) , as_type<float>(944879383u) , as_type<float>(947628162u) , as_type<float>(949144823u) , as_type<float>(950519212u) , as_type<float>(951893602u) , as_type<float>(953267991u) , as_type<float>(961656599u) , as_type<float>(966609234u) , as_type<float>(970045207u) , as_type<float>(973279855u) , as_type<float>(974997842u) , as_type<float>(976715828u) , as_type<float>(978433815u) , as_type<float>(980151802u) , as_type<float>(981668463u) , as_type<float>(990057071u) , as_type<float>(994352038u) , as_type<float>(998445679u) , as_type<float>(1000593162u) , as_type<float>(1002740646u) , as_type<float>(1004888130u) , as_type<float>(1006834287u) , as_type<float>(1007908028u) , as_type<float>(1008981770u) }
;

constant float tc_r_i [ 64 ] = {
as_type<float>(786163455u) , as_type<float>(794552063u) , as_type<float>(799337791u) , as_type<float>(802940671u) , as_type<float>(805924959u) , as_type<float>(807726399u) , as_type<float>(809527839u) , as_type<float>(811329279u) , as_type<float>(813130719u) , as_type<float>(814313567u) , as_type<float>(822702175u) , as_type<float>(827205775u) , as_type<float>(831090783u) , as_type<float>(833342583u) , as_type<float>(835594383u) , as_type<float>(837846183u) , as_type<float>(839479391u) , as_type<float>(840605291u) , as_type<float>(841731191u) , as_type<float>(850119799u) , as_type<float>(855693657u) , as_type<float>(858508407u) , as_type<float>(861323157u) , as_type<float>(864082265u) , as_type<float>(865489640u) , as_type<float>(866897015u) , as_type<float>(868304390u) , as_type<float>(869711765u) , as_type<float>(878100373u) , as_type<float>(882970544u) , as_type<float>(886488981u) , as_type<float>(889599933u) , as_type<float>(891359152u) , as_type<float>(893118370u) , as_type<float>(894877589u) , as_type<float>(896636807u) , as_type<float>(897988541u) , as_type<float>(906377149u) , as_type<float>(910775196u) , as_type<float>(914765757u) , as_type<float>(916964780u) , as_type<float>(919163804u) , as_type<float>(921362827u) , as_type<float>(923154365u) , as_type<float>(924253877u) , as_type<float>(925353388u) , as_type<float>(933741996u) , as_type<float>(939239554u) , as_type<float>(942130604u) , as_type<float>(944879383u) , as_type<float>(947628162u) , as_type<float>(949144823u) , as_type<float>(950519212u) , as_type<float>(951893602u) , as_type<float>(953267991u) , as_type<float>(961656599u) , as_type<float>(966609234u) , as_type<float>(970045207u) , as_type<float>(973279855u) , as_type<float>(974997842u) , as_type<float>(976715828u) , as_type<float>(978433815u) , as_type<float>(980151802u) , as_type<float>(981668463u) }
;

constant float tc_r_g [ 28 ] = {
as_type<float>(925353388u) , as_type<float>(933741996u) , as_type<float>(939239554u) , as_type<float>(942130604u) , as_type<float>(944879383u) , as_type<float>(947628162u) , as_type<float>(949144823u) , as_type<float>(950519212u) , as_type<float>(951893602u) , as_type<float>(953267991u) , as_type<float>(961656599u) , as_type<float>(966609234u) , as_type<float>(970045207u) , as_type<float>(973279855u) , as_type<float>(974997842u) , as_type<float>(976715828u) , as_type<float>(978433815u) , as_type<float>(980151802u) , as_type<float>(981668463u) , as_type<float>(990057071u) , as_type<float>(994352038u) , as_type<float>(998445679u) , as_type<float>(1000593162u) , as_type<float>(1002740646u) , as_type<float>(1004888130u) , as_type<float>(1006834287u) , as_type<float>(1007908028u) , as_type<float>(1008981770u) }
;

constant float tc_r_c [ 37 ] = {
as_type<float>(897988541u) , as_type<float>(906377149u) , as_type<float>(910775196u) , as_type<float>(914765757u) , as_type<float>(916964780u) , as_type<float>(919163804u) , as_type<float>(921362827u) , as_type<float>(923154365u) , as_type<float>(924253877u) , as_type<float>(925353388u) , as_type<float>(933741996u) , as_type<float>(939239554u) , as_type<float>(942130604u) , as_type<float>(944879383u) , as_type<float>(947628162u) , as_type<float>(949144823u) , as_type<float>(950519212u) , as_type<float>(951893602u) , as_type<float>(953267991u) , as_type<float>(961656599u) , as_type<float>(966609234u) , as_type<float>(970045207u) , as_type<float>(973279855u) , as_type<float>(974997842u) , as_type<float>(976715828u) , as_type<float>(978433815u) , as_type<float>(980151802u) , as_type<float>(981668463u) , as_type<float>(990057071u) , as_type<float>(994352038u) , as_type<float>(998445679u) , as_type<float>(1000593162u) , as_type<float>(1002740646u) , as_type<float>(1004888130u) , as_type<float>(1006834287u) , as_type<float>(1007908028u) , as_type<float>(1008981770u) }
;

constant float tc_ocg2 [ 15 ] = {
as_type<float>(1026206379u) , as_type<float>(1007192200u) , as_type<float>(985008992u) , as_type<float>(961547516u) , as_type<float>(936381698u) , as_type<float>(909700894u) , as_type<float>(882111097u) , as_type<float>(852963883u) , as_type<float>(823097026u) , as_type<float>(791712302u) , as_type<float>(759810980u) , as_type<float>(727138212u) , as_type<float>(693583780u) , as_type<float>(659199547u) , as_type<float>(624169905u) }
;

constant float tc_ocg1 [ 15 ] = {
as_type<float>(1065353216u) , as_type<float>(1056964608u) , as_type<float>(1042983595u) , as_type<float>(1026206379u) , as_type<float>(1007192200u) , as_type<float>(985008992u) , as_type<float>(961547516u) , as_type<float>(936381698u) , as_type<float>(909700894u) , as_type<float>(882111097u) , as_type<float>(852963883u) , as_type<float>(823097026u) , as_type<float>(791712302u) , as_type<float>(759810980u) , as_type<float>(727138212u) }
;

constant float tc_nt_in [ 55 ] = {
as_type<float>(1065353216u) , as_type<float>(1073741824u) , as_type<float>(1077936128u) , as_type<float>(1082130432u) , as_type<float>(1084227584u) , as_type<float>(1086324736u) , as_type<float>(1088421888u) , as_type<float>(1090519040u) , as_type<float>(1091567616u) , as_type<float>(1092616192u) , as_type<float>(1101004800u) , as_type<float>(1106247680u) , as_type<float>(1109393408u) , as_type<float>(1112014848u) , as_type<float>(1114636288u) , as_type<float>(1116471296u) , as_type<float>(1117782016u) , as_type<float>(1119092736u) , as_type<float>(1120403456u) , as_type<float>(1128792064u) , as_type<float>(1133903872u) , as_type<float>(1137180672u) , as_type<float>(1140457472u) , as_type<float>(1142292480u) , as_type<float>(1143930880u) , as_type<float>(1145569280u) , as_type<float>(1147207680u) , as_type<float>(1148846080u) , as_type<float>(1157234688u) , as_type<float>(1161527296u) , as_type<float>(1165623296u) , as_type<float>(1167867904u) , as_type<float>(1169915904u) , as_type<float>(1171963904u) , as_type<float>(1174011904u) , as_type<float>(1175232512u) , as_type<float>(1176256512u) , as_type<float>(1184645120u) , as_type<float>(1189765120u) , as_type<float>(1193033728u) , as_type<float>(1195593728u) , as_type<float>(1198153728u) , as_type<float>(1200142336u) , as_type<float>(1201422336u) , as_type<float>(1202702336u) , as_type<float>(1203982336u) , as_type<float>(1212370944u) , as_type<float>(1217559552u) , as_type<float>(1220759552u) , as_type<float>(1223959552u) , as_type<float>(1225948160u) , as_type<float>(1227548160u) , as_type<float>(1229148160u) , as_type<float>(1230748160u) , as_type<float>(1232348160u) }
;

constant float tc_nt_i [ 55 ] = {
as_type<float>(1065353216u) , as_type<float>(1073741824u) , as_type<float>(1077936128u) , as_type<float>(1082130432u) , as_type<float>(1084227584u) , as_type<float>(1086324736u) , as_type<float>(1088421888u) , as_type<float>(1090519040u) , as_type<float>(1091567616u) , as_type<float>(1092616192u) , as_type<float>(1101004800u) , as_type<float>(1106247680u) , as_type<float>(1109393408u) , as_type<float>(1112014848u) , as_type<float>(1114636288u) , as_type<float>(1116471296u) , as_type<float>(1117782016u) , as_type<float>(1119092736u) , as_type<float>(1120403456u) , as_type<float>(1128792064u) , as_type<float>(1133903872u) , as_type<float>(1137180672u) , as_type<float>(1140457472u) , as_type<float>(1142292480u) , as_type<float>(1143930880u) , as_type<float>(1145569280u) , as_type<float>(1147207680u) , as_type<float>(1148846080u) , as_type<float>(1157234688u) , as_type<float>(1161527296u) , as_type<float>(1165623296u) , as_type<float>(1167867904u) , as_type<float>(1169915904u) , as_type<float>(1171963904u) , as_type<float>(1174011904u) , as_type<float>(1175232512u) , as_type<float>(1176256512u) , as_type<float>(1184645120u) , as_type<float>(1189765120u) , as_type<float>(1193033728u) , as_type<float>(1195593728u) , as_type<float>(1198153728u) , as_type<float>(1200142336u) , as_type<float>(1201422336u) , as_type<float>(1202702336u) , as_type<float>(1203982336u) , as_type<float>(1212370944u) , as_type<float>(1217559552u) , as_type<float>(1220759552u) , as_type<float>(1223959552u) , as_type<float>(1225948160u) , as_type<float>(1227548160u) , as_type<float>(1229148160u) , as_type<float>(1230748160u) , as_type<float>(1232348160u) }
;

constant Soft8 tc_ds [ 100 ] = {
SD8(0x3f2ad333ee243328ul) , SD8(0x3f2c16d81a78be12ul) , SD8(0x3f2d69bcfaade57eul) , SD8(0x3f2ecc9a94a4d8f2ul) , SD8(0x3f302018cd3b94beul) , SD8(0x3f30e2a5e98cc55cul) , SD8(0x3f31ae5e43f67fbaul) , SD8(0x3f3283b07bbcf881ul) , SD8(0x3f33631066c9bf1dul) , SD8(0x3f344cf750921f39ul) , SD8(0x3f3541e43bf463eeul) , SD8(0x3f36425c282fc07dul) , SD8(0x3f374eea591c5216ul) , SD8(0x3f386820a2ca71b1ul) , SD8(0x3f398e97b8b26644ul) , SD8(0x3f3ac2ef809f7722ul) , SD8(0x3f3c05cf69836519ul) , SD8(0x3f3d57e6c6616feful) , SD8(0x3f3eb9ed2d8246e4ul) , SD8(0x3f4016516e124b35ul) , SD8(0x3f40d8688f702a7eul) , SD8(0x3f41a3a55f7abf18ul) , SD8(0x3f4278763a60cb63ul) , SD8(0x3f43574eafcd1076ul) , SD8(0x3f4440a7c1a68ab6ul) , SD8(0x3f45350025c5c3faul) , SD8(0x3f4634dc8ac2e968ul) , SD8(0x3f4740c7e00201a6ul) , SD8(0x3f485953a12462aful) , SD8(0x3f497f1825085ee5ul) , SD8(0x3f4ab2b4f0820ee2ul) , SD8(0x3f4bf4d10cfa2349ul) , SD8(0x3f4d461b6321c7cbul) , SD8(0x3f4ea74b19ecd7eful) , SD8(0x3f500c8ffd02fc4bul) , SD8(0x3f50ce316af94b94ul) , SD8(0x3f5198f2fb8fadacul) , SD8(0x3f526d42c8082108ul) , SD8(0x3f534b9419f9318eul) , SD8(0x3f54345fa9e8277eul) , SD8(0x3f552823e0d680caul) , SD8(0x3f5627651ce64d49ul) , SD8(0x3f5732adf93ab3f1ul) , SD8(0x3f584a8f993ba89bul) , SD8(0x3f596fa1f765b0edul) , SD8(0x3f5aa28437d0841dul) , SD8(0x3f5be3dcfe995567ul) , SD8(0x3f5d345aca5fb610ul) , SD8(0x3f5e94b453063179ul) , SD8(0x3f6002d476750b85ul) , SD8(0x3f60c40078642915ul) , SD8(0x3f618e471443dd61ul) , SD8(0x3f6262162091f8a0ul) , SD8(0x3f633fe0a0fb50eful) , SD8(0x3f64281f04cffa83ul) , SD8(0x3f651b4f6869004cul) , SD8(0x3f6619f5d9a32002ul) , SD8(0x3f67249c9f93b8cdul) , SD8(0x3f683bd4859edcabul) , SD8(0x3f6960352a174a50ul) , SD8(0x3f6a925d50930130ul) , SD8(0x3f6bd2f3382124a1ul) , SD8(0x3f6d22a4f58ffe0eul) , SD8(0x3f6e8228d1f42349ul) , SD8(0x3f6ff23dada4156cul) , SD8(0x3f70b9d5b3ef0bfful) , SD8(0x3f7183a1a5a84432ul) , SD8(0x3f7256f03fdfd262ul) , SD8(0x3f73343440833c64ul) , SD8(0x3f741be5cdd9c775ul) , SD8(0x3f750e82b7c2884ful) , SD8(0x3f760c8ebc05981bul) , SD8(0x3f771693cddd870ful) , SD8(0x3f782d2260dfe51eul) , SD8(0x3f7950d1b76d8dd2ul) , SD8(0x3f7a824034d54f12ul) , SD8(0x3f7bc213b3558474ul) , SD8(0x3f7d10f9de2b5b08ul) , SD8(0x3f7e6fa88fe0a4d5ul) , SD8(0x3f7fdede350b7137ul) , SD8(0x3f80afb119da8498ul) , SD8(0x3f817902abd03ae2ul) , SD8(0x3f824bd121d5adf1ul) , SD8(0x3f83288ef4435f40ul) , SD8(0x3f840fb400840f41ul) , SD8(0x3f8501bdca2b3cf4ul) , SD8(0x3f85ff2fbf1cecf9ul) , SD8(0x3f8708937eebba8eul) , SD8(0x3f881e792593f31ful) , SD8(0x3f89417799bc50faul) , SD8(0x3f8a722cdea6d303ul) , SD8(0x3f8bb13e69fe301aul) , SD8(0x3f8cff597dae7daaul) , SD8(0x3f8e5d3385f9d2eeul) , SD8(0x3f8fcb8a7bfbff3eul) , SD8(0x3f90a592a66968e2ul) , SD8(0x3f916e6a22d17b5bul) , SD8(0x3f9240b8c25a08f0ul) , SD8(0x3f931cf0b7f0c0eful) , SD8(0x3f94038998500e50ul) }
;

constant Soft8 tc_dr [ 100 ] = {
SD8(0x3f0ad333ee243328ul) , SD8(0x3f0c16d81a78be10ul) , SD8(0x3f0d69bcfaade57dul) , SD8(0x3f0ecc9a94a4d8f0ul) , SD8(0x3f102018cd3b94bdul) , SD8(0x3f10e2a5e98cc55bul) , SD8(0x3f11ae5e43f67fb8ul) , SD8(0x3f1283b07bbcf880ul) , SD8(0x3f13631066c9bf1cul) , SD8(0x3f144cf750921f38ul) , SD8(0x3f1541e43bf463edul) , SD8(0x3f16425c282fc07cul) , SD8(0x3f174eea591c5215ul) , SD8(0x3f186820a2ca71b0ul) , SD8(0x3f198e97b8b26640ul) , SD8(0x3f1ac2ef809f7721ul) , SD8(0x3f1c05cf6983651bul) , SD8(0x3f1d57e6c6616feeul) , SD8(0x3f1eb9ed2d8246e0ul) , SD8(0x3f2016516e124b34ul) , SD8(0x3f20d8688f702a7ful) , SD8(0x3f21a3a55f7abf17ul) , SD8(0x3f2278763a60cb60ul) , SD8(0x3f23574eafcd1071ul) , SD8(0x3f2440a7c1a68ab2ul) , SD8(0x3f25350025c5c3f9ul) , SD8(0x3f2634dc8ac2e967ul) , SD8(0x3f2740c7e00201a8ul) , SD8(0x3f285953a12462b4ul) , SD8(0x3f297f1825085ee7ul) , SD8(0x3f2ab2b4f0820ee0ul) , SD8(0x3f2bf4d10cfa2344ul) , SD8(0x3f2d461b6321c7c9ul) , SD8(0x3f2ea74b19ecd7f2ul) , SD8(0x3f300c8ffd02fc4aul) , SD8(0x3f30ce316af94b93ul) , SD8(0x3f3198f2fb8fada9ul) , SD8(0x3f326d42c8082105ul) , SD8(0x3f334b9419f9318dul) , SD8(0x3f34345fa9e8277dul) , SD8(0x3f352823e0d680c6ul) , SD8(0x3f3627651ce64d45ul) , SD8(0x3f3732adf93ab3f1ul) , SD8(0x3f384a8f993ba89aul) , SD8(0x3f396fa1f765b0ebul) , SD8(0x3f3aa28437d0841bul) , SD8(0x3f3be3dcfe995566ul) , SD8(0x3f3d345aca5fb60eul) , SD8(0x3f3e94b453063177ul) , SD8(0x3f4002d476750b84ul) , SD8(0x3f40c40078642914ul) , SD8(0x3f418e471443dd5ful) , SD8(0x3f4262162091f89ful) , SD8(0x3f433fe0a0fb50eeul) , SD8(0x3f44281f04cffa82ul) , SD8(0x3f451b4f6869004bul) , SD8(0x3f4619f5d9a32001ul) , SD8(0x3f47249c9f93b8cbul) , SD8(0x3f483bd4859edcaaul) , SD8(0x3f4960352a174a4eul) , SD8(0x3f4a925d5093012ful) , SD8(0x3f4bd2f3382124a0ul) , SD8(0x3f4d22a4f58ffe0dul) , SD8(0x3f4e8228d1f42348ul) , SD8(0x3f4ff23dada4156bul) , SD8(0x3f50b9d5b3ef0bfful) , SD8(0x3f5183a1a5a84431ul) , SD8(0x3f5256f03fdfd261ul) , SD8(0x3f53343440833c64ul) , SD8(0x3f541be5cdd9c774ul) , SD8(0x3f550e82b7c2884eul) , SD8(0x3f560c8ebc05981aul) , SD8(0x3f571693cddd870eul) , SD8(0x3f582d2260dfe51cul) , SD8(0x3f5950d1b76d8dd0ul) , SD8(0x3f5a824034d54f0ful) , SD8(0x3f5bc213b3558472ul) , SD8(0x3f5d10f9de2b5b06ul) , SD8(0x3f5e6fa88fe0a4d3ul) , SD8(0x3f5fdede350b7135ul) , SD8(0x3f60afb119da8497ul) , SD8(0x3f617902abd03ae1ul) , SD8(0x3f624bd121d5adeful) , SD8(0x3f63288ef4435f3eul) , SD8(0x3f640fb400840f40ul) , SD8(0x3f6501bdca2b3cf3ul) , SD8(0x3f65ff2fbf1cecf8ul) , SD8(0x3f6708937eebba8cul) , SD8(0x3f681e792593f31dul) , SD8(0x3f69417799bc50f9ul) , SD8(0x3f6a722cdea6d302ul) , SD8(0x3f6bb13e69fe3019ul) , SD8(0x3f6cff597dae7da9ul) , SD8(0x3f6e5d3385f9d2edul) , SD8(0x3f6fcb8a7bfbff3dul) , SD8(0x3f70a592a66968e1ul) , SD8(0x3f716e6a22d17b5aul) , SD8(0x3f7240b8c25a08eful) , SD8(0x3f731cf0b7f0c0eeul) , SD8(0x3f74038998500e4ful) }
;

constant float tc_csg [ 18 ] = {
as_type<float>(1073741824u) , as_type<float>(1086324736u) , as_type<float>(1086324736u) , as_type<float>(1080088863u) , as_type<float>(1113783747u) , as_type<float>(1103101952u) , as_type<float>(1081591910u) , as_type<float>(1096890194u) , as_type<float>(1115848985u) , as_type<float>(1089707021u) , as_type<float>(1126514157u) , as_type<float>(1115848985u) , as_type<float>(1068521978u) , as_type<float>(1068521978u) , as_type<float>(1063642469u) , as_type<float>(1064094685u) , as_type<float>(1078001471u) , as_type<float>(1089707021u) }
;

constant float tc_cse [ 18 ] = {
as_type<float>(1077936128u) , as_type<float>(1082130432u) , as_type<float>(1082130432u) , as_type<float>(1080242995u) , as_type<float>(1085381018u) , as_type<float>(1084227584u) , as_type<float>(1080602447u) , as_type<float>(1083463592u) , as_type<float>(1085560744u) , as_type<float>(1082519873u) , as_type<float>(1086714178u) , as_type<float>(1085560744u) , as_type<float>(1076048691u) , as_type<float>(1076048691u) , as_type<float>(1070685854u) , as_type<float>(1071854387u) , as_type<float>(1079658728u) , as_type<float>(1082519873u) }
;

constant float tc_crg [ 13 ] = {
as_type<float>(1086324736u) , as_type<float>(1065353216u) , as_type<float>(1086324736u) , as_type<float>(1144258561u) , as_type<float>(1065353216u) , as_type<float>(1103101952u) , as_type<float>(1079292361u) , as_type<float>(1144258561u) , as_type<float>(1086324736u) , as_type<float>(1065353216u) , as_type<float>(1073741824u) , as_type<float>(1068115923u) , as_type<float>(1167949828u) }
;

constant float tc_cre [ 13 ] = {
as_type<float>(1082130432u) , as_type<float>(1065353216u) , as_type<float>(1082130432u) , as_type<float>(1088421888u) , as_type<float>(1073741824u) , as_type<float>(1084227584u) , as_type<float>(1080033280u) , as_type<float>(1088421888u) , as_type<float>(1082130432u) , as_type<float>(1073741824u) , as_type<float>(1077936128u) , as_type<float>(1075838976u) , as_type<float>(1090519040u) }
;

constant float tc_cig [ 7 ] = {
as_type<float>(1065353216u) , as_type<float>(1086324736u) , as_type<float>(1103101952u) , as_type<float>(1065353216u) , as_type<float>(1065353216u) , as_type<float>(1079292361u) , as_type<float>(1068115923u) }
;

constant float tc_cie [ 7 ] = {
as_type<float>(1065353216u) , as_type<float>(1082130432u) , as_type<float>(1084227584u) , as_type<float>(1073741824u) , as_type<float>(1073741824u) , as_type<float>(1080033280u) , as_type<float>(1075838976u) }
;

constant float tc_cgg [ 12 ] = {
as_type<float>(1086324736u) , as_type<float>(1065353216u) , as_type<float>(1086324736u) , as_type<float>(1144258561u) , as_type<float>(1165791270u) , as_type<float>(1101195236u) , as_type<float>(1120347906u) , as_type<float>(1142071356u) , as_type<float>(1084719917u) , as_type<float>(1065353216u) , as_type<float>(1072921172u) , as_type<float>(1072921172u) }
;

constant float tc_cge [ 12 ] = {
as_type<float>(1082130432u) , as_type<float>(1065353216u) , as_type<float>(1082130432u) , as_type<float>(1088421888u) , as_type<float>(1090288353u) , as_type<float>(1083996897u) , as_type<float>(1086094049u) , as_type<float>(1088191201u) , as_type<float>(1081669058u) , as_type<float>(1073741824u) , as_type<float>(1077705441u) , as_type<float>(1077705441u) }
;

constant float tc_ccg [ 75 ] = {
as_type<float>(1065353216u) , as_type<float>(1103101952u) , as_type<float>(1167949828u) , as_type<float>(1086324736u) , as_type<float>(1144258561u) , as_type<float>(1073741824u) , as_type<float>(1123024897u) , as_type<float>(1193115647u) , as_type<float>(1103101952u) , as_type<float>(1167949828u) , as_type<float>(1086324736u) , as_type<float>(1144258561u) , as_type<float>(1219571711u) , as_type<float>(1123024897u) , as_type<float>(1193115647u) , as_type<float>(1103101952u) , as_type<float>(1167949828u) , as_type<float>(1247640583u) , as_type<float>(1144258561u) , as_type<float>(1219571711u) , as_type<float>(1123024897u) , as_type<float>(1193115647u) , as_type<float>(1276659008u) , as_type<float>(1167949828u) , as_type<float>(1247640583u) , as_type<float>(1144258561u) , as_type<float>(1219571711u) , as_type<float>(1306814440u) , as_type<float>(1193115647u) , as_type<float>(1276659008u) , as_type<float>(1167949828u) , as_type<float>(1247640583u) , as_type<float>(1337562217u) , as_type<float>(1219571711u) , as_type<float>(1306814440u) , as_type<float>(1193115647u) , as_type<float>(1276659008u) , as_type<float>(1369596378u) , as_type<float>(1247640583u) , as_type<float>(1337562217u) , as_type<float>(1219571711u) , as_type<float>(1306814440u) , as_type<float>(1402485688u) , as_type<float>(1276659008u) , as_type<float>(1369596378u) , as_type<float>(1247640583u) , as_type<float>(1337562217u) , as_type<float>(1436040120u) , as_type<float>(1306814440u) , as_type<float>(1402485688u) , as_type<float>(1276659008u) , as_type<float>(1369596378u) , as_type<float>(1470218104u) , as_type<float>(1337562217u) , as_type<float>(1436040120u) , as_type<float>(1306814440u) , as_type<float>(1402485688u) , as_type<float>(1505097593u) , as_type<float>(1369596378u) , as_type<float>(1470218104u) , as_type<float>(1337562217u) , as_type<float>(1436040120u) , as_type<float>(1540888031u) , as_type<float>(1402485688u) , as_type<float>(1505097593u) , as_type<float>(1369596378u) , as_type<float>(1470218104u) , as_type<float>(1577520547u) , as_type<float>(1436040120u) , as_type<float>(1540888031u) , as_type<float>(1402485688u) , as_type<float>(1505097593u) , as_type<float>(1613840857u) , as_type<float>(1470218104u) , as_type<float>(1577520547u) }
;

constant float tc_cce [ 75 ] = {
as_type<float>(1073741824u) , as_type<float>(1084227584u) , as_type<float>(1090519040u) , as_type<float>(1082130432u) , as_type<float>(1088421888u) , as_type<float>(1077936128u) , as_type<float>(1086324736u) , as_type<float>(1091567616u) , as_type<float>(1084227584u) , as_type<float>(1090519040u) , as_type<float>(1082130432u) , as_type<float>(1088421888u) , as_type<float>(1092616192u) , as_type<float>(1086324736u) , as_type<float>(1091567616u) , as_type<float>(1084227584u) , as_type<float>(1090519040u) , as_type<float>(1093664768u) , as_type<float>(1088421888u) , as_type<float>(1092616192u) , as_type<float>(1086324736u) , as_type<float>(1091567616u) , as_type<float>(1094713344u) , as_type<float>(1090519040u) , as_type<float>(1093664768u) , as_type<float>(1088421888u) , as_type<float>(1092616192u) , as_type<float>(1095761920u) , as_type<float>(1091567616u) , as_type<float>(1094713344u) , as_type<float>(1090519040u) , as_type<float>(1093664768u) , as_type<float>(1096810496u) , as_type<float>(1092616192u) , as_type<float>(1095761920u) , as_type<float>(1091567616u) , as_type<float>(1094713344u) , as_type<float>(1097859072u) , as_type<float>(1093664768u) , as_type<float>(1096810496u) , as_type<float>(1092616192u) , as_type<float>(1095761920u) , as_type<float>(1098907648u) , as_type<float>(1094713344u) , as_type<float>(1097859072u) , as_type<float>(1093664768u) , as_type<float>(1096810496u) , as_type<float>(1099431936u) , as_type<float>(1095761920u) , as_type<float>(1098907648u) , as_type<float>(1094713344u) , as_type<float>(1097859072u) , as_type<float>(1099956224u) , as_type<float>(1096810496u) , as_type<float>(1099431936u) , as_type<float>(1095761920u) , as_type<float>(1098907648u) , as_type<float>(1100480512u) , as_type<float>(1097859072u) , as_type<float>(1099956224u) , as_type<float>(1096810496u) , as_type<float>(1099431936u) , as_type<float>(1101004800u) , as_type<float>(1098907648u) , as_type<float>(1100480512u) , as_type<float>(1097859072u) , as_type<float>(1099956224u) , as_type<float>(1101529088u) , as_type<float>(1099431936u) , as_type<float>(1101004800u) , as_type<float>(1098907648u) , as_type<float>(1100480512u) , as_type<float>(1102053376u) , as_type<float>(1099956224u) , as_type<float>(1101529088u) }
;
constant float am_g = as_type<float>(1132652115u) ;
constant float am_i = as_type<float>(1139343456u) ;
constant float am_r = as_type<float>(1141040723u) ;
constant float am_s = as_type<float>(1032671199u) ;
constant float ar_volume = as_type<float>(613870276u) ;
constant float ato = as_type<float>(1050387939u) ;
constant float av_c = as_type<float>(1274138500u) ;
constant float av_g = as_type<float>(1138556928u) ;
constant float av_i = as_type<float>(1155985408u) ;
constant float av_r = as_type<float>(1167568896u) ;
constant float av_s = as_type<float>(1109393408u) ;
constant float bm_g = as_type<float>(1077936128u) ;
constant float bm_i = as_type<float>(1077936128u) ;
constant float bm_r = as_type<float>(1077936128u) ;
constant float bm_s = as_type<float>(1073741824u) ;
constant float bv_c = as_type<float>(1073741824u) ;
constant float bv_g = as_type<float>(1063507722u) ;
constant float bv_i = as_type<float>(1065353216u) ;
constant float bv_s = as_type<float>(1057803469u) ;
constant float c_cube = as_type<float>(1056964608u) ;
constant float c_sqrd = as_type<float>(1041865114u) ;
constant float cp = as_type<float>(1148911616u) ;
constant float d0c = as_type<float>(897988541u) ;
constant float d0g = as_type<float>(964891247u) ;
constant float d0i = as_type<float>(928540250u) ;
constant float d0r = as_type<float>(944879383u) ;
constant float d0s = as_type<float>(961656599u) ;
constant bool dustyice = true ;
constant float ef_ri = as_type<float>(1064514355u) ;
constant float ef_si = as_type<float>(1028443341u) ;
constant float eps = as_type<float>(646978941u) ;
constant float fv_r = as_type<float>(1128464384u) ;
constant float fv_s = as_type<float>(1120403456u) ;
constant float gonv_max = as_type<float>(1245125376u) ;
constant float gonv_min = as_type<float>(1176256512u) ;
constant float hgfr = as_type<float>(1131096310u) ;
constant bool homogice = true ;
constant int hostinclusionstring_g = 0 ;
constant int hostinclusionstring_s = 0 ;
constant int hostmatrixstring_g = 0 ;
constant int hostmatrixstring_s = 0 ;
constant int hoststring_g = 0 ;
constant int hoststring_s = 0 ;
constant int ifdry = 0 ;
constant bool iiwarm = false ;
constant int inclusionstring_g = 0 ;
constant int inclusionstring_s = 0 ;
constant bool is_aerosol_aware = false ;
constant Soft8 k_w = SD8(0x3fede4b5a79d9de5ul) ;
constant float kap0 = as_type<float>(1140149453u) ;
constant float kap1 = as_type<float>(1099673108u) ;
constant float lam0 = as_type<float>(1101413745u) ;
constant float lam1 = as_type<float>(1079152476u) ;
constant Soft8 lamda4 = SD8(0x3f1a36e305532621ul) ;
constant Soft8 lamda_radar = SD8(0x3fb99999a0000000ul) ;
constant float lfus = as_type<float>(1218647552u) ;
constant float lsub = as_type<float>(1244461376u) ;
constant float lvap0 = as_type<float>(1243125376u) ;
constant int matrixstring_g = 0 ;
constant int matrixstring_s = 0 ;
constant Soft8 melt_outside_g = SD8(0x3feccccccccccccdul) ;
constant Soft8 melt_outside_s = SD8(0x3feccccccccccccdul) ;
constant int mixingrulestring_g = 0 ;
constant int mixingrulestring_s = 0 ;
constant float mu_g = as_type<float>(0u) ;
constant float mu_i = as_type<float>(0u) ;
constant float mu_r = as_type<float>(0u) ;
constant float mu_s = as_type<float>(1059241276u) ;
constant float nain1 = as_type<float>(1223959552u) ;
constant int nbc = 100 ;
constant int nbins = 100 ;
constant int nbr = 100 ;
constant int nbs = 100 ;
constant int nic1 = 7 ;
constant int nic2 = - 6 ;
constant int nig2 = - 5 ;
constant int nig3 = 4 ;
constant int nii2 = - 10 ;
constant int nii3 = 0 ;
constant int niin2 = 0 ;
constant int nir2 = - 6 ;
constant int nir3 = 6 ;
constant int nis2 = - 5 ;
constant int nrbins = 50 ;
constant float nt_c_max = as_type<float>(1324240036u) ;
constant int ntb_arc = 7 ;
constant int ntb_art = 7 ;
constant int ntb_arw = 9 ;
constant int ntb_c = 37 ;
constant int ntb_g = 28 ;
constant int ntb_g1 = 28 ;
constant int ntb_i = 64 ;
constant int ntb_i1 = 55 ;
constant int ntb_in = 55 ;
constant int ntb_r = 37 ;
constant int ntb_r1 = 37 ;
constant int ntb_s = 28 ;
constant int ntb_t = 9 ;
constant float oams = as_type<float>(1097327186u) ;
constant float obmg = as_type<float>(1051372203u) ;
constant float obmi = as_type<float>(1051372203u) ;
constant float obmr = as_type<float>(1051372203u) ;
constant float obms = as_type<float>(1056964608u) ;
constant float ocmg = as_type<float>(1042289142u) ;
constant float ocms = as_type<float>(1081320659u) ;
constant float oge1 = as_type<float>(1048576000u) ;
constant float ogg1 = as_type<float>(1042983595u) ;
constant float ogg2 = as_type<float>(1065353216u) ;
constant float ogg3 = as_type<float>(1042983595u) ;
constant float oig1 = as_type<float>(1065353216u) ;
constant float oig2 = as_type<float>(1042983595u) ;
constant float olfus = as_type<float>(910748860u) ;
constant float org1 = as_type<float>(1042983595u) ;
constant float org2 = as_type<float>(1065353216u) ;
constant float org3 = as_type<float>(1042983595u) ;
constant float orv = as_type<float>(990773674u) ;
constant float pi = as_type<float>(1078530011u) ;
constant Soft8 pi5 = SD8(0x4073204ba0000000ul) ;
constant float r = as_type<float>(1133479199u) ;
constant float r1 = as_type<float>(730643660u) ;
constant float r2 = as_type<float>(897988541u) ;
constant float r_uni = as_type<float>(1090848293u) ;
constant float rho_g = as_type<float>(1140457472u) ;
constant float rho_i = as_type<float>(1147043840u) ;
constant float rho_not = as_type<float>(1066901091u) ;
constant float rho_w = as_type<float>(1148846080u) ;
constant float rv = as_type<float>(1139195904u) ;
constant float sc = as_type<float>(1059179201u) ;
constant float t1_qg_me = as_type<float>(931640973u) ;
constant float t1_qg_qc = as_type<float>(1155737411u) ;
constant float t1_qg_sd = as_type<float>(1063004406u) ;
constant float t1_qr_ev = as_type<float>(1061662228u) ;
constant float t1_qr_qc = as_type<float>(1186116576u) ;
constant float t1_qr_qi = as_type<float>(1186116576u) ;
constant float t1_qs_me = as_type<float>(916642577u) ;
constant float t1_qs_qc = as_type<float>(1106990034u) ;
constant float t1_qs_qi = as_type<float>(1106990034u) ;
constant float t1_qs_sd = as_type<float>(1063004406u) ;
constant float t2_qg_me = as_type<float>(960335067u) ;
constant float t2_qg_sd = as_type<float>(1092206496u) ;
constant float t2_qr_ev = as_type<float>(1108562439u) ;
constant float t2_qr_qi = as_type<float>(1319851067u) ;
constant float t2_qs_me = as_type<float>(923788342u) ;
constant float t2_qs_sd = as_type<float>(1069712850u) ;
constant float t_0 = as_type<float>(1133024051u) ;
constant float tno = as_type<float>(1084227584u) ;
constant float xm0i = as_type<float>(730643660u) ;
struct ThompsonColumn {
device F * tc_wsf ;
device Soft8 * tc_wsd ;
device bool * tc_wsb ;
int nz ;
F mu_c ;
constant F * cce = (constant F*)tc_cce ;
constant F * ccg = (constant F*)tc_ccg ;
constant F * cge = (constant F*)tc_cge ;
constant F * cgg = (constant F*)tc_cgg ;
constant F * cie = (constant F*)tc_cie ;
constant F * cig = (constant F*)tc_cig ;
constant F * cre = (constant F*)tc_cre ;
constant F * crg = (constant F*)tc_crg ;
constant F * cse = (constant F*)tc_cse ;
constant F * csg = (constant F*)tc_csg ;
constant Soft8 * dr = tc_dr ;
constant Soft8 * ds = tc_ds ;
cd m_i_0 = {
SD8(0x3ffc91dae74d3b24ul) , - SD8(0x3f21acb622ae704dul) }
;
cd m_w_0 = {
SD8(0x402213d1a0000000ul) , - SD8(0x3ff634d6c0000000ul) }
;
F nt_c ;
constant F * nt_i = (constant F*)tc_nt_i ;
constant F * nt_in = (constant F*)tc_nt_in ;
constant F * ocg1 = (constant F*)tc_ocg1 ;
constant F * ocg2 = (constant F*)tc_ocg2 ;
constant F * r_c = (constant F*)tc_r_c ;
constant F * r_g = (constant F*)tc_r_g ;
constant F * r_i = (constant F*)tc_r_i ;
constant F * r_r = (constant F*)tc_r_r ;
constant F * r_s = (constant F*)tc_r_s ;
constant F * sa = (constant F*)tc_sa ;
constant F * sb = (constant F*)tc_sb ;
constant Soft8 * simpson = tc_simpson ;
device const Soft8 * t_efrw ;
device const Soft8 * t_efsw ;
constant Soft8 * t_nc = tc_t_nc ;
constant F * ta_na = (constant F*)tc_ta_na ;
constant F * ta_tk = (constant F*)tc_ta_tk ;
constant F * ta_ww = (constant F*)tc_ta_ww ;
device const Soft8 * tcg_racg ;
device const Soft8 * tcr_gacr ;
device const Soft8 * tcr_sacr1 ;
device const Soft8 * tcr_sacr2 ;
device const Soft8 * tcs_racs1 ;
device const Soft8 * tcs_racs2 ;
device const Soft8 * tmg_gacr ;
device const Soft8 * tmr_racg ;
device const Soft8 * tmr_racs1 ;
device const Soft8 * tmr_racs2 ;
device const Soft8 * tms_sacr1 ;
device const Soft8 * tms_sacr2 ;
device const Soft8 * tnc_wev ;
device const F * tnccn_act ;
device const Soft8 * tni_iaus ;
device const Soft8 * tni_qcfz ;
device const Soft8 * tni_qrfz ;
device const Soft8 * tnr_gacr ;
device const Soft8 * tnr_qrfz ;
device const Soft8 * tnr_racg ;
device const Soft8 * tnr_racs1 ;
device const Soft8 * tnr_racs2 ;
device const Soft8 * tnr_rev ;
device const Soft8 * tnr_sacr1 ;
device const Soft8 * tnr_sacr2 ;
device const Soft8 * tpc_wev ;
device const Soft8 * tpg_qrfz ;
device const Soft8 * tpi_ide ;
device const Soft8 * tpi_qcfz ;
device const Soft8 * tpi_qrfz ;
device const Soft8 * tps_iaus ;
constant Soft8 * xdtg = tc_xdtg ;
constant Soft8 * xdts = tc_xdts ;
constant Soft8 * xxdg = tc_xxdg ;
constant Soft8 * xxds = tc_xxds ;
void rayleigh_soak_wetgraupel (Soft8 x, Soft8 a, Soft8 b, Soft8 fm, Soft8 mo, cd mw, cd mi, Soft8 lambda, thread Soft8 & cback, int, int, int, int, int, int) {
cback = refl_rayleigh_soak_wetgraupel ( x , a , b , fm , mo , mw , mi , pi5 , lamda4 ) ;
}
void calc_refl10cm (TCSPACE F * qv1d, TCSPACE F * qc1d, TCSPACE F * qr1d, TCSPACE F * nr1d, TCSPACE F * qs1d, TCSPACE F * qg1d, TCSPACE F * t1d, TCSPACE F * p1d, TCSPACE F * dbz, int kts, int kte, int ii, int jj) {
TCF ( temp , TCKMAX ) ;
TCF ( pres , TCKMAX ) ;
TCF ( qv , TCKMAX ) ;
TCF ( rho , TCKMAX ) ;
TCF ( rhof , TCKMAX ) ;
TCF ( rc , TCKMAX ) ;
TCF ( rr , TCKMAX ) ;
TCF ( nr , TCKMAX ) ;
TCF ( rs , TCKMAX ) ;
TCF ( rg , TCKMAX ) ;
TCD ( ilamr , TCKMAX ) ;
TCD ( ilamg , TCKMAX ) ;
TCD ( n0_r , TCKMAX ) ;
TCD ( n0_g , TCKMAX ) ;
TCF ( mvd_r , TCKMAX ) ;
TCF ( smob , TCKMAX ) ;
TCF ( smo2 , TCKMAX ) ;
TCF ( smoc , TCKMAX ) ;
TCF ( smoz , TCKMAX ) ;
F om3 ;
F m0 ;
F mrat ;
F slam1 ;
F slam2 ;
F xds ;
F ils1 ;
F ils2 ;
F t1_vts ;
F t2_vts ;
F t3_vts ;
F t4_vts ;
F vtr_dbz_wt ;
F vts_dbz_wt ;
F vtg_dbz_wt ;
TCF ( ze_rain , TCKMAX ) ;
TCF ( ze_snow , TCKMAX ) ;
TCF ( ze_graupel , TCKMAX ) ;
Soft8 n0_exp ;
Soft8 n0_min ;
Soft8 lam_exp ;
Soft8 lamr ;
Soft8 lamg ;
F a_ ;
F b_ ;
F loga_ ;
F tc0 ;
Soft8 fmelt_s ;
Soft8 fmelt_g ;
int i ;
int k ;
int k_0 ;
int kbot ;
int n ;
bool melti ;
TCB ( l_qr , TCKMAX ) ;
TCB ( l_qs , TCKMAX ) ;
TCB ( l_qg , TCKMAX ) ;
Soft8 cback ;
Soft8 x ;
Soft8 eta ;
Soft8 f_d ;
F xslw1 ;
F ygra1 ;
F zans1 ;
for ( k = kts ;
k <= kte ;
k += 1 ) {
dbz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = ( - 35.0f ) ;
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = t1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ;
qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 1.e-10f , qv1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = p1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ;
rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( 0.622f , pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_mul ( tc_mul ( r , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 0.622f ) ) ) ) ) ;
rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( tc_div ( rho_not , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_mul ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
if ( ( qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r2 , tc_mul ( nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
lamr = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_r , crg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , org2 ) , nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( 1.f , lamr ) ;
n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , org2 ) , tc_pow ( lamr , cre [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) ;
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
else {
rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 50.e-6f ;
l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r2 ) ) {
rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
else {
rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r2 ) ) {
rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
else {
rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smob [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smoc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smoz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
}
if ( tc_any ( l_qs , kts , kte ) ) {
for ( k = kts ;
k <= kte ;
k += 1 ) {
if ( ( ! l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) {
continue ;
}
tc0 = tc_min ( ( - 0.1f ) , tc_sub ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) ) ;
smob [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , oams ) ;
if ( ( ( bm_s > ( tc_sub ( 2.0f , 1.e-3f ) ) ) && ( bm_s < ( tc_add ( 2.0f , 1.e-3f ) ) ) ) ) {
smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = smob [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ;
}
else {
loga_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sa [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , bm_s ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , bm_s ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , bm_s ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , bm_s ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , bm_s ) , bm_s ) , bm_s ) ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sb [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , bm_s ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , bm_s ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , bm_s ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , bm_s ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , bm_s ) , bm_s ) , bm_s ) ) ;
smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_pow ( ( tc_div ( smob [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , a_ ) ) , ( tc_div ( 1.f , b_ ) ) ) ;
}
loga_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sa [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sb [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ;
smoc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( a_ , tc_pow ( smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , b_ ) ) ;
loga_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sa [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sb [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) ;
smoz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( a_ , tc_pow ( smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , b_ ) ) ;
}
}
if ( tc_any ( l_qg , kts , kte ) ) {
n0_min = gonv_max ;
k_0 = kts ;
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
if ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] >= 270.65f ) ) {
k_0 = tc_max ( k_0 , k ) ;
}
}
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
if ( ( ( ( k > k_0 ) && l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) && ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 100.e-6f ) ) ) {
xslw1 = tc_add ( 4.01f , tc_log10 ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
}
else {
xslw1 = 0.01f ;
}
ygra1 = tc_add ( 4.31f , tc_log10 ( tc_max ( 5.e-5f , rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
zans1 = tc_add ( 3.1f , ( tc_div ( 100.f , ( tc_add ( tc_add ( tc_div ( tc_mul ( tc_mul ( 300.f , xslw1 ) , ygra1 ) , ( tc_add ( tc_add ( tc_div ( 10.f , xslw1 ) , 1.f ) , tc_mul ( 0.25f , ygra1 ) ) ) ) , 30.f ) , tc_mul ( 10.f , ygra1 ) ) ) ) ) ) ;
n0_exp = tc_pow ( 10.f , ( zans1 ) ) ;
n0_exp = tc_max ( ( ( Soft8 ) ( gonv_min ) ) , tc_min ( n0_exp , ( ( Soft8 ) ( gonv_max ) ) ) ) ;
n0_min = tc_min ( n0_exp , n0_min ) ;
n0_exp = n0_min ;
lam_exp = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( n0_exp , am_g ) , cgg [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , oge1 ) ;
lamg = tc_mul ( lam_exp , tc_pow ( ( tc_mul ( tc_mul ( cgg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , ogg2 ) , ogg1 ) ) , obmg ) ) ;
ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( 1.f , lamg ) ;
n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_div ( n0_exp , ( tc_mul ( cgg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , lam_exp ) ) ) , tc_pow ( lamg , cge [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) ;
}
}
melti = false ;
k_0 = kts ;
for ( k = tc_sub ( kte , 1 ) ;
k >= kts ;
k += ( - 1 ) ) {
if ( ( ( ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 273.15f ) ) && l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) && ( ( l_qs [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] || l_qg [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] ) ) ) ) {
k_0 = tc_max ( tc_add ( k , 1 ) , k_0 ) ;
melti = true ;
break ;
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
ze_rain [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 1.e-22f ;
ze_snow [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 1.e-22f ;
ze_graupel [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 1.e-22f ;
if ( l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) {
ze_rain [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , crg [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cre [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) ) ;
}
if ( l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) {
ze_snow [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( ( tc_div ( 0.176f , 0.93f ) ) , ( tc_div ( 6.0f , pi ) ) ) , ( tc_div ( 6.0f , pi ) ) ) , ( tc_div ( am_s , 900.0f ) ) ) , ( tc_div ( am_s , 900.0f ) ) ) , smoz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
if ( l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) {
ze_graupel [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( ( tc_div ( 0.176f , 0.93f ) ) , ( tc_div ( 6.0f , pi ) ) ) , ( tc_div ( 6.0f , pi ) ) ) , ( tc_div ( am_g , 900.0f ) ) ) , ( tc_div ( am_g , 900.0f ) ) ) , n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , cgg [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) ) ;
}
}
if ( ( ( ( ! iiwarm ) && melti ) && ( k_0 >= 2 ) ) ) {
for ( k = tc_sub ( k_0 , 1 ) ;
k >= kts ;
k += ( - 1 ) ) {
if ( ( l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] && l_qs [ 0 + tc_sub ( k_0 , ( kts ) ) * ( 1 ) ] ) ) {
fmelt_s = tc_max ( SD8(0x3fa999999999999aul) , tc_min ( tc_sub ( SD8(0x3ff0000000000000ul) , tc_div ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rs [ 0 + tc_sub ( k_0 , ( kts ) ) * ( 1 ) ] ) ) , SD8(0x3fefae147ae147aeul) ) ) ;
eta = SD8(0x0000000000000000ul) ;
om3 = tc_div ( 1.f , smoc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
m0 = ( tc_mul ( smob [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , om3 ) ) ;
mrat = tc_mul ( tc_mul ( tc_mul ( smob [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , m0 ) , m0 ) , m0 ) ;
slam1 = tc_mul ( m0 , lam0 ) ;
slam2 = tc_mul ( m0 , lam1 ) ;
for ( n = 1 ;
n <= nrbins ;
n += 1 ) {
x = tc_mul ( am_s , tc_pow ( xxds [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] , bm_s ) ) ;
rayleigh_soak_wetgraupel ( x , ( ( Soft8 ) ( ocms ) ) , ( ( Soft8 ) ( obms ) ) , fmelt_s , melt_outside_s , m_w_0 , m_i_0 , lamda_radar , cback , mixingrulestring_s , matrixstring_s , inclusionstring_s , hoststring_s , hostmatrixstring_s , hostinclusionstring_s ) ;
f_d = tc_mul ( mrat , ( tc_add ( tc_mul ( kap0 , tc_exp ( tc_mul ( ( - slam1 ) , xxds [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] ) ) ) , tc_mul ( tc_mul ( kap1 , tc_pow ( ( tc_mul ( m0 , xxds [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] ) ) , mu_s ) ) , tc_exp ( tc_mul ( ( - slam2 ) , xxds [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ) ;
eta = tc_add ( eta , tc_mul ( tc_mul ( tc_mul ( f_d , cback ) , simpson [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] ) , xdts [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] ) ) ;
}
ze_snow [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sngl ( tc_mul ( tc_div ( lamda4 , ( tc_mul ( pi5 , k_w ) ) ) , eta ) ) ;
}
if ( ( l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] && l_qg [ 0 + tc_sub ( k_0 , ( kts ) ) * ( 1 ) ] ) ) {
fmelt_g = tc_max ( SD8(0x3fa999999999999aul) , tc_min ( tc_sub ( SD8(0x3ff0000000000000ul) , tc_div ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rg [ 0 + tc_sub ( k_0 , ( kts ) ) * ( 1 ) ] ) ) , SD8(0x3fefae147ae147aeul) ) ) ;
eta = SD8(0x0000000000000000ul) ;
lamg = tc_div ( 1.f , ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
for ( n = 1 ;
n <= nrbins ;
n += 1 ) {
x = tc_mul ( am_g , tc_pow ( xxdg [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] , bm_g ) ) ;
rayleigh_soak_wetgraupel ( x , ( ( Soft8 ) ( ocmg ) ) , ( ( Soft8 ) ( obmg ) ) , fmelt_g , melt_outside_g , m_w_0 , m_i_0 , lamda_radar , cback , mixingrulestring_g , matrixstring_g , inclusionstring_g , hoststring_g , hostmatrixstring_g , hostinclusionstring_g ) ;
f_d = tc_mul ( tc_mul ( n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( xxdg [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] , mu_g ) ) , tc_exp ( tc_mul ( ( - lamg ) , xxdg [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] ) ) ) ;
eta = tc_add ( eta , tc_mul ( tc_mul ( tc_mul ( f_d , cback ) , simpson [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] ) , xdtg [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] ) ) ;
}
ze_graupel [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sngl ( tc_mul ( tc_div ( lamda4 , ( tc_mul ( pi5 , k_w ) ) ) , eta ) ) ;
}
}
}
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
dbz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( 10.f , tc_log10 ( tc_mul ( ( tc_add ( tc_add ( ze_rain [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ze_snow [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ze_graupel [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , SD8(0x43abc16d674ec800ul) ) ) ) ;
}
}
}
;
#include "hexv842_mpas_thompson_column_radar_f32_bindings.metalh"
kernel void hexv842_mpas_thompson_column_radar_f32(
    constant Hexv842MpasThompsonColumnRadarF32Resources &r [[buffer(0)]],
    constant Hexv842MpasThompsonColumnRadarF32Params &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    using namespace ntc;
    device const F *qv = (device F *)r.qv;
    device const F *qc = (device F *)r.qc;
    device const F *qr = (device F *)r.qr;
    device const F *nr = (device F *)r.nr;
    device const F *qs = (device F *)r.qs;
    device const F *qg = (device F *)r.qg;
    device const F *th = (device F *)r.th;
    device const F *pii = (device F *)r.pii;
    device const F *p = (device F *)r.p;
    device const F *z = (device F *)r.z;
    device const F *dz = (device F *)r.dz;
    device F *refl = (device F *)r.refl;
    device F *refl_max = (device F *)r.refl_max;
    device F *refl_1km = (device F *)r.refl_1km;
    device F *refl_1km_max = (device F *)r.refl_1km_max;
    const int nz = params.nz;
    const int columns = params.columns;
{
int slot = int(gid); if(slot>=params.count)return; int col=params.first+slot;
if ( col >= columns ) return ;
ThompsonColumn m; m.nz=nz; m.tc_wsf=(device F*)r.ws_f+ulong(slot)*28*(nz+1); m.tc_wsd=(device Soft8*)r.ws_d+ulong(slot)*4*(nz+1); m.tc_wsb=(device bool*)r.ws_b+ulong(slot)*3*(nz+1);
TCKF ( qv1 , TCKMAX ) ; TCKF ( qc1 , TCKMAX ) ; TCKF ( qr1 , TCKMAX ) ;
TCKF ( nr1 , TCKMAX ) ; TCKF ( qs1 , TCKMAX ) ; TCKF ( qg1 , TCKMAX ) ;
TCKF ( t1 , TCKMAX ) ; TCKF ( p1 , TCKMAX ) ; TCKF ( dbz , TCKMAX ) ; TCKF ( zp , TCKMAX ) ;
for ( int k = 0 ;
k < nz ;
k ++ ) {
int idx = k * columns + col ;
qv1 [ k ] = qv [ idx ] ;
qc1 [ k ] = qc [ idx ] ;
qr1 [ k ] = qr [ idx ] ;
nr1 [ k ] = nr [ idx ] ;
qs1 [ k ] = qs [ idx ] ;
qg1 [ k ] = qg [ idx ] ;
t1 [ k ] = tc_mul ( th [ idx ] , pii [ idx ] ) ;
p1 [ k ] = p [ idx ] ;
dbz [ k ] = - 35.0f ;
zp [ k ] = tc_add ( tc_sub ( z [ idx ] , z [ col ] ) , tc_mul ( 0.5f , dz [ idx ] ) ) ;
}
m . calc_refl10cm ( qv1 , qc1 , qr1 , nr1 , qs1 , qg1 , t1 , p1 , dbz , 1 , nz , col + 1 , 1 ) ;
int kp = 0 ;
F peak = - 35.0f ;
for ( int k = 0 ;
k < nz ;
k ++ ) {
dbz [ k ] = tc_max ( - 35.0f , dbz [ k ] ) ;
refl [ k * columns + col ] = dbz [ k ] ;
peak = tc_max ( peak , dbz [ k ] ) ;
if ( zp [ k ] < 1000.0f ) kp = k ;
}
refl_max [ col ] = peak ;
F w1 = tc_div ( ( tc_sub ( zp [ kp + 1 ] , 1000.0f ) ) , ( tc_sub ( zp [ kp + 1 ] , zp [ kp ] ) ) ) ;
F w2 = tc_sub ( 1.0f , w1 ) ;
refl_1km [ col ] = tc_add ( tc_mul ( w1 , dbz [ kp ] ) , tc_mul ( w2 , dbz [ kp + 1 ] ) ) ;
refl_1km_max [ col ] = tc_max ( refl_1km_max [ col ] , refl_1km [ col ] ) ;
}

}