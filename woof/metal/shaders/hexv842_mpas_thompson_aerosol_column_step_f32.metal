#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
using namespace ntc;
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
constant bool is_aerosol_aware = true ;
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
F rslf (F p, F t) {
F result ;
F esl ;
F x ;
F c0 = .611583699e03f ;
F c1 = .444606896e02f ;
F c2 = .143177157e01f ;
F c3 = .264224321e-1f ;
F c4 = .299291081e-3f ;
F c5 = .203154182e-5f ;
F c6 = .702620698e-8f ;
F c7 = .379534310e-11f ;
F c8 = ( - .321582393e-13f ) ;
x = tc_max ( ( - 80.f ) , tc_sub ( t , 273.16f ) ) ;
esl = tc_add ( c0 , tc_mul ( x , ( tc_add ( c1 , tc_mul ( x , ( tc_add ( c2 , tc_mul ( x , ( tc_add ( c3 , tc_mul ( x , ( tc_add ( c4 , tc_mul ( x , ( tc_add ( c5 , tc_mul ( x , ( tc_add ( c6 , tc_mul ( x , ( tc_add ( c7 , tc_mul ( x , c8 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ;
esl = tc_min ( esl , tc_mul ( p , 0.15f ) ) ;
result = tc_div ( tc_mul ( .622f , esl ) , ( tc_sub ( p , esl ) ) ) ;
return result ;
}
F rsif (F p, F t) {
F result ;
F esi ;
F x ;
F c0 = .609868993e03f ;
F c1 = .499320233e02f ;
F c2 = .184672631e01f ;
F c3 = .402737184e-1f ;
F c4 = .565392987e-3f ;
F c5 = .521693933e-5f ;
F c6 = .307839583e-7f ;
F c7 = .105785160e-9f ;
F c8 = .161444444e-12f ;
x = tc_max ( ( - 80.f ) , tc_sub ( t , 273.16f ) ) ;
esi = tc_add ( c0 , tc_mul ( x , ( tc_add ( c1 , tc_mul ( x , ( tc_add ( c2 , tc_mul ( x , ( tc_add ( c3 , tc_mul ( x , ( tc_add ( c4 , tc_mul ( x , ( tc_add ( c5 , tc_mul ( x , ( tc_add ( c6 , tc_mul ( x , ( tc_add ( c7 , tc_mul ( x , c8 ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ;
esi = tc_min ( esi , tc_mul ( p , 0.15f ) ) ;
result = tc_div ( tc_mul ( .622f , esi ) , ( tc_sub ( p , esi ) ) ) ;
return result ;
}
F icedemott (F tempc, F qv, F qvs, F qvsi, F rho, F nifa) {
F result ;
F satw ;
F sati ;
F siw ;
F p_x ;
F si0x ;
F dtt ;
F dsi ;
F dsw ;
F dab ;
F fc ;
F hx ;
F ntilde ;
F n_in ;
F nmax ;
F nhat ;
F mux ;
F xni ;
F nifa_cc ;
F p_c1 = 1000.f ;
F p_rho_c = 0.76f ;
F p_alpha = 1.0f ;
F p_gam = 2.f ;
F delt = 5.f ;
F t0x = ( - 40.f ) ;
F sw0x = 0.97f ;
F delsi = 0.1f ;
F hdm = 0.15f ;
F p_psi = tc_div ( tc_mul ( 0.058707f , p_gam ) , p_rho_c ) ;
F aap = 1.f ;
F bbp = 0.f ;
F y1p = ( - 35.f ) ;
F y2p = ( - 25.f ) ;
F rho_not0 = tc_div ( 101325.f , ( tc_mul ( 287.05f , 273.15f ) ) ) ;
xni = 0.0f ;
nifa_cc = tc_max ( 0.5f , tc_div ( tc_mul ( tc_mul ( nifa , rho_not0 ) , 1.e-6f ) , rho ) ) ;
xni = tc_mul ( ( tc_mul ( 5.94e-5f , tc_pow ( ( ( - tempc ) ) , 3.33f ) ) ) , ( tc_pow ( nifa_cc , ( tc_add ( ( tc_mul ( ( - 0.0264f ) , ( tempc ) ) ) , 0.0033f ) ) ) ) ) ;
xni = tc_mul ( tc_div ( tc_mul ( xni , rho ) , rho_not0 ) , 1000.f ) ;
result = tc_max ( 0.f , xni ) ;
return result ;
}
F icekoop (F temp, F qv, F qvs, F naero, F dt) {
F result ;
F mu_diff ;
F a_w_i ;
F delta_aw ;
F log_j_rate ;
F j_rate ;
F prob_h ;
F satw ;
F xni ;
xni = 0.0f ;
satw = tc_div ( qv , qvs ) ;
mu_diff = tc_sub ( tc_sub ( tc_add ( 210368.0f , ( tc_mul ( 131.438f , temp ) ) ) , ( tc_div ( 3.32373e6f , temp ) ) ) , ( tc_mul ( 41729.1f , tc_log ( temp ) ) ) ) ;
a_w_i = tc_exp ( tc_div ( mu_diff , ( tc_mul ( r_uni , temp ) ) ) ) ;
delta_aw = tc_sub ( satw , a_w_i ) ;
log_j_rate = tc_add ( tc_sub ( tc_add ( ( - 906.7f ) , ( tc_mul ( 8502.0f , delta_aw ) ) ) , ( tc_mul ( tc_mul ( 26924.0f , delta_aw ) , delta_aw ) ) ) , ( tc_mul ( tc_mul ( tc_mul ( 29180.0f , delta_aw ) , delta_aw ) , delta_aw ) ) ) ;
log_j_rate = tc_min ( 20.0f , log_j_rate ) ;
j_rate = tc_pow ( 10.f , log_j_rate ) ;
prob_h = tc_min ( tc_sub ( 1.f , tc_exp ( tc_mul ( tc_mul ( ( - j_rate ) , ar_volume ) , dt ) ) ) , 1.f ) ;
if ( ( prob_h > 0.f ) ) {
xni = tc_min ( tc_mul ( prob_h , naero ) , 1000.e3f ) ;
}
result = tc_max ( 0.0f , xni ) ;
return result ;
}
F delta_p (F yy, F y1, F y2, F aa, F bb) {
F result ;
F dab ;
F a ;
F b ;
F a0 ;
F a1 ;
F a2 ;
F a3 ;
a = tc_div ( tc_mul ( 6.f , ( tc_sub ( aa , bb ) ) ) , ( tc_mul ( tc_mul ( ( tc_sub ( y2 , y1 ) ) , ( tc_sub ( y2 , y1 ) ) ) , ( tc_sub ( y2 , y1 ) ) ) ) ) ;
b = tc_sub ( tc_add ( aa , ( tc_div ( tc_mul ( tc_mul ( tc_mul ( a , y1 ) , y1 ) , y1 ) , 6.f ) ) ) , ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( a , y1 ) , y1 ) , y2 ) , 0.5f ) ) ) ;
a0 = b ;
a1 = tc_mul ( tc_mul ( a , y1 ) , y2 ) ;
a2 = tc_mul ( tc_mul ( ( - a ) , ( tc_add ( y1 , y2 ) ) ) , 0.5f ) ;
a3 = tc_div ( a , 3.f ) ;
if ( ( yy <= y1 ) ) {
dab = aa ;
}
else if ( ( yy >= y2 ) ) {
dab = bb ;
}
else {
dab = tc_add ( tc_add ( tc_add ( a0 , ( tc_mul ( a1 , yy ) ) ) , ( tc_mul ( tc_mul ( a2 , yy ) , yy ) ) ) , ( tc_mul ( tc_mul ( tc_mul ( a3 , yy ) , yy ) , yy ) ) ) ;
}
if ( ( dab < aa ) ) {
dab = aa ;
}
if ( ( dab > bb ) ) {
dab = bb ;
}
result = dab ;
return result ;
}
F eff_aero (F d, F da, F visc, F rhoa, F temp, int species) {
F result ;
F aval ;
F cc ;
F diff ;
F re ;
F sc ;
F st ;
F st2 ;
F vt ;
F eff ;
F boltzman = 1.3806503e-23f ;
F meanpath = 0.0256e-6f ;
vt = 1.f ;
if ( ( species == 114 ) ) {
vt = tc_sub ( tc_add ( tc_sub ( tc_add ( ( - 0.1021f ) , tc_mul ( 4.932e3f , d ) ) , tc_mul ( tc_mul ( 0.9551e6f , d ) , d ) ) , tc_mul ( tc_mul ( tc_mul ( 0.07934e9f , d ) , d ) , d ) ) , tc_mul ( tc_mul ( tc_mul ( tc_mul ( 0.002362e12f , d ) , d ) , d ) , d ) ) ;
}
else if ( ( species == 115 ) ) {
vt = tc_mul ( av_s , tc_pow ( d , bv_s ) ) ;
}
else if ( ( species == 103 ) ) {
vt = tc_mul ( av_g , tc_pow ( d , bv_g ) ) ;
}
cc = tc_add ( 1.f , tc_mul ( tc_div ( tc_mul ( 2.f , meanpath ) , da ) , ( tc_add ( 1.257f , tc_mul ( 0.4f , tc_exp ( tc_div ( tc_mul ( ( - 0.55f ) , da ) , meanpath ) ) ) ) ) ) ) ;
diff = tc_div ( tc_mul ( tc_mul ( boltzman , temp ) , cc ) , ( tc_mul ( tc_mul ( tc_mul ( 3.f , pi ) , visc ) , da ) ) ) ;
re = tc_div ( tc_mul ( tc_mul ( tc_mul ( 0.5f , rhoa ) , d ) , vt ) , visc ) ;
sc = tc_div ( visc , ( tc_mul ( rhoa , diff ) ) ) ;
st = tc_div ( tc_mul ( tc_mul ( tc_mul ( da , da ) , vt ) , 1000.f ) , ( tc_mul ( tc_mul ( 9.f , visc ) , d ) ) ) ;
aval = tc_add ( 1.f , tc_log ( tc_add ( 1.f , re ) ) ) ;
st2 = tc_div ( ( tc_add ( 1.2f , tc_mul ( tc_div ( 1.f , 12.f ) , aval ) ) ) , ( tc_add ( 1.f , aval ) ) ) ;
eff = tc_add ( tc_mul ( tc_div ( 4.f , ( tc_mul ( re , sc ) ) ) , ( tc_add ( tc_add ( 1.f , tc_mul ( tc_mul ( 0.4f , tc_sqrt ( re ) ) , tc_pow ( sc , 0.3333f ) ) ) , tc_mul ( tc_mul ( 0.16f , tc_sqrt ( re ) ) , tc_sqrt ( sc ) ) ) ) ) , tc_mul ( tc_div ( tc_mul ( 4.f , da ) , d ) , ( tc_add ( 0.02f , tc_mul ( tc_div ( da , d ) , ( tc_add ( 1.f , tc_mul ( 2.f , tc_sqrt ( re ) ) ) ) ) ) ) ) ) ;
if ( ( st > st2 ) ) {
eff = tc_add ( eff , tc_pow ( ( tc_div ( ( tc_sub ( st , st2 ) ) , ( tc_add ( tc_sub ( st , st2 ) , 0.666667f ) ) ) ) , 1.5f ) ) ;
}
result = tc_max ( 1.e-5f , tc_min ( eff , 1.0f ) ) ;
return result ;
}
F activ_ncloud (F tt, F ww, F nccn) {
F result ;
F n_local ;
F w_local ;
int i ;
int j ;
int k ;
int l ;
int m ;
int n ;
F a ;
F b ;
F c ;
F d ;
F t ;
F u ;
F x1 ;
F x2 ;
F y1 ;
F y2 ;
F nx ;
F wy ;
F fraction ;
n_local = tc_mul ( nccn , 1.e-6f ) ;
w_local = ww ;
if ( ( n_local >= ta_na [ 0 + tc_sub ( ntb_arc , ( 1 ) ) * ( 1 ) ] ) ) {
n_local = tc_sub ( ta_na [ 0 + tc_sub ( ntb_arc , ( 1 ) ) * ( 1 ) ] , 1.0f ) ;
}
else if ( ( n_local <= ta_na [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
n_local = tc_add ( ta_na [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , 1.0f ) ;
}
for ( n = 2 ;
n <= ntb_arc ;
n += 1 ) {
if ( ( ( n_local >= ta_na [ 0 + tc_sub ( tc_sub ( n , 1 ) , ( 1 ) ) * ( 1 ) ] ) && ( n_local < ta_na [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] ) ) ) {
break ;
}
}
i = n ;
x1 = tc_log ( ta_na [ 0 + tc_sub ( tc_sub ( i , 1 ) , ( 1 ) ) * ( 1 ) ] ) ;
x2 = tc_log ( ta_na [ 0 + tc_sub ( i , ( 1 ) ) * ( 1 ) ] ) ;
if ( ( w_local >= ta_ww [ 0 + tc_sub ( ntb_arw , ( 1 ) ) * ( 1 ) ] ) ) {
w_local = tc_sub ( ta_ww [ 0 + tc_sub ( ntb_arw , ( 1 ) ) * ( 1 ) ] , 1.0f ) ;
}
else if ( ( w_local <= ta_ww [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
w_local = tc_add ( ta_ww [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , 0.001f ) ;
}
for ( n = 2 ;
n <= ntb_arw ;
n += 1 ) {
if ( ( ( w_local >= ta_ww [ 0 + tc_sub ( tc_sub ( n , 1 ) , ( 1 ) ) * ( 1 ) ] ) && ( w_local < ta_ww [ 0 + tc_sub ( n , ( 1 ) ) * ( 1 ) ] ) ) ) {
break ;
}
}
j = n ;
y1 = tc_log ( ta_ww [ 0 + tc_sub ( tc_sub ( j , 1 ) , ( 1 ) ) * ( 1 ) ] ) ;
y2 = tc_log ( ta_ww [ 0 + tc_sub ( j , ( 1 ) ) * ( 1 ) ] ) ;
k = tc_max ( 1 , tc_min ( tc_add ( ( ( int ) round ( tc_mul ( ( tc_sub ( tt , ta_tk [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , 0.1f ) ) ) , 1 ) , ntb_art ) ) ;
l = 3 ;
m = 2 ;
a = tnccn_act [ 0 + tc_sub ( tc_sub ( i , 1 ) , ( 1 ) ) * ( 1 ) + tc_sub ( tc_sub ( j , 1 ) , ( 1 ) ) * tc_mul ( 1 , ( ( 7 ) - ( 1 ) + 1 ) ) + tc_sub ( k , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( l , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 7 ) - ( 1 ) + 1 ) ) + tc_sub ( m , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 5 ) - ( 1 ) + 1 ) ) ] ;
b = tnccn_act [ 0 + tc_sub ( i , ( 1 ) ) * ( 1 ) + tc_sub ( tc_sub ( j , 1 ) , ( 1 ) ) * tc_mul ( 1 , ( ( 7 ) - ( 1 ) + 1 ) ) + tc_sub ( k , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( l , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 7 ) - ( 1 ) + 1 ) ) + tc_sub ( m , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 5 ) - ( 1 ) + 1 ) ) ] ;
c = tnccn_act [ 0 + tc_sub ( i , ( 1 ) ) * ( 1 ) + tc_sub ( j , ( 1 ) ) * tc_mul ( 1 , ( ( 7 ) - ( 1 ) + 1 ) ) + tc_sub ( k , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( l , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 7 ) - ( 1 ) + 1 ) ) + tc_sub ( m , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 5 ) - ( 1 ) + 1 ) ) ] ;
d = tnccn_act [ 0 + tc_sub ( tc_sub ( i , 1 ) , ( 1 ) ) * ( 1 ) + tc_sub ( j , ( 1 ) ) * tc_mul ( 1 , ( ( 7 ) - ( 1 ) + 1 ) ) + tc_sub ( k , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( l , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 7 ) - ( 1 ) + 1 ) ) + tc_sub ( m , ( 1 ) ) * ( 1 * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 7 ) - ( 1 ) + 1 ) * ( ( 5 ) - ( 1 ) + 1 ) ) ] ;
nx = tc_log ( n_local ) ;
wy = tc_log ( w_local ) ;
t = tc_div ( ( tc_sub ( nx , x1 ) ) , ( tc_sub ( x2 , x1 ) ) ) ;
u = tc_div ( ( tc_sub ( wy , y1 ) ) , ( tc_sub ( y2 , y1 ) ) ) ;
fraction = tc_add ( tc_add ( tc_add ( tc_mul ( tc_mul ( ( tc_sub ( 1.0f , t ) ) , ( tc_sub ( 1.0f , u ) ) ) , a ) , tc_mul ( tc_mul ( t , ( tc_sub ( 1.0f , u ) ) ) , b ) ) , tc_mul ( tc_mul ( t , u ) , c ) ) , tc_mul ( tc_mul ( ( tc_sub ( 1.0f , t ) ) , u ) , d ) ) ;
result = tc_mul ( nccn , fraction ) ;
return result ;
}
void mp_thompson (TCSPACE F * qv1d, TCSPACE F * qc1d, TCSPACE F * qi1d, TCSPACE F * qr1d, TCSPACE F * qs1d, TCSPACE F * qg1d, TCSPACE F * ni1d, TCSPACE F * nr1d, TCSPACE F * nc1d, TCSPACE F * nwfa1d, TCSPACE F * nifa1d, TCSPACE F * t1d, TCSPACE F * p1d, TCSPACE F * w1d, TCSPACE F * dzq, thread F & pptrain, thread F & pptsnow, thread F & pptgraul, thread F & pptice, TCSPACE F * rainprod, TCSPACE F * evapprod, int kts, int kte, F dt, int ii, int jj) {
TCF ( tten , TCKMAX ) ;
TCF ( qvten , TCKMAX ) ;
TCF ( qcten , TCKMAX ) ;
TCF ( qiten , TCKMAX ) ;
TCF ( qrten , TCKMAX ) ;
TCF ( qsten , TCKMAX ) ;
TCF ( qgten , TCKMAX ) ;
TCF ( niten , TCKMAX ) ;
TCF ( nrten , TCKMAX ) ;
TCF ( ncten , TCKMAX ) ;
TCF ( nwfaten , TCKMAX ) ;
TCF ( nifaten , TCKMAX ) ;
TCD ( prw_vcd , TCKMAX ) ;
TCD ( pnc_wcd , TCKMAX ) ;
TCD ( pnc_wau , TCKMAX ) ;
TCD ( pnc_rcw , TCKMAX ) ;
TCD ( pnc_scw , TCKMAX ) ;
TCD ( pnc_gcw , TCKMAX ) ;
TCD ( pna_rca , TCKMAX ) ;
TCD ( pna_sca , TCKMAX ) ;
TCD ( pna_gca , TCKMAX ) ;
TCD ( pnd_rcd , TCKMAX ) ;
TCD ( pnd_scd , TCKMAX ) ;
TCD ( pnd_gcd , TCKMAX ) ;
TCD ( prr_wau , TCKMAX ) ;
TCD ( prr_rcw , TCKMAX ) ;
TCD ( prr_rcs , TCKMAX ) ;
TCD ( prr_rcg , TCKMAX ) ;
TCD ( prr_sml , TCKMAX ) ;
TCD ( prr_gml , TCKMAX ) ;
TCD ( prr_rci , TCKMAX ) ;
TCD ( prv_rev , TCKMAX ) ;
TCD ( pnr_wau , TCKMAX ) ;
TCD ( pnr_rcs , TCKMAX ) ;
TCD ( pnr_rcg , TCKMAX ) ;
TCD ( pnr_rci , TCKMAX ) ;
TCD ( pnr_sml , TCKMAX ) ;
TCD ( pnr_gml , TCKMAX ) ;
TCD ( pnr_rev , TCKMAX ) ;
TCD ( pnr_rcr , TCKMAX ) ;
TCD ( pnr_rfz , TCKMAX ) ;
TCD ( pri_inu , TCKMAX ) ;
TCD ( pni_inu , TCKMAX ) ;
TCD ( pri_ihm , TCKMAX ) ;
TCD ( pni_ihm , TCKMAX ) ;
TCD ( pri_wfz , TCKMAX ) ;
TCD ( pni_wfz , TCKMAX ) ;
TCD ( pri_rfz , TCKMAX ) ;
TCD ( pni_rfz , TCKMAX ) ;
TCD ( pri_ide , TCKMAX ) ;
TCD ( pni_ide , TCKMAX ) ;
TCD ( pri_rci , TCKMAX ) ;
TCD ( pni_rci , TCKMAX ) ;
TCD ( pni_sci , TCKMAX ) ;
TCD ( pni_iau , TCKMAX ) ;
TCD ( pri_iha , TCKMAX ) ;
TCD ( pni_iha , TCKMAX ) ;
TCD ( prs_iau , TCKMAX ) ;
TCD ( prs_sci , TCKMAX ) ;
TCD ( prs_rcs , TCKMAX ) ;
TCD ( prs_scw , TCKMAX ) ;
TCD ( prs_sde , TCKMAX ) ;
TCD ( prs_ihm , TCKMAX ) ;
TCD ( prs_ide , TCKMAX ) ;
TCD ( prg_scw , TCKMAX ) ;
TCD ( prg_rfz , TCKMAX ) ;
TCD ( prg_gde , TCKMAX ) ;
TCD ( prg_gcw , TCKMAX ) ;
TCD ( prg_rci , TCKMAX ) ;
TCD ( prg_rcs , TCKMAX ) ;
TCD ( prg_rcg , TCKMAX ) ;
TCD ( prg_ihm , TCKMAX ) ;
Soft8 zerod0 = SD8(0x0000000000000000ul) ;
TCF ( temp , TCKMAX ) ;
TCF ( pres , TCKMAX ) ;
TCF ( qv , TCKMAX ) ;
TCF ( rc , TCKMAX ) ;
TCF ( ri , TCKMAX ) ;
TCF ( rr , TCKMAX ) ;
TCF ( rs , TCKMAX ) ;
TCF ( rg , TCKMAX ) ;
TCF ( ni , TCKMAX ) ;
TCF ( nr , TCKMAX ) ;
TCF ( nc , TCKMAX ) ;
TCF ( nwfa , TCKMAX ) ;
TCF ( nifa , TCKMAX ) ;
TCF ( rho , TCKMAX ) ;
TCF ( rhof , TCKMAX ) ;
TCF ( rhof2 , TCKMAX ) ;
TCF ( qvs , TCKMAX ) ;
TCF ( qvsi , TCKMAX ) ;
TCF ( delqvs , TCKMAX ) ;
TCF ( satw , TCKMAX ) ;
TCF ( sati , TCKMAX ) ;
TCF ( ssatw , TCKMAX ) ;
TCF ( ssati , TCKMAX ) ;
TCF ( diffu , TCKMAX ) ;
TCF ( visco , TCKMAX ) ;
TCF ( vsc2 , TCKMAX ) ;
TCF ( tcond , TCKMAX ) ;
TCF ( lvap , TCKMAX ) ;
TCF ( ocp , TCKMAX ) ;
TCF ( lvt2 , TCKMAX ) ;
TCD ( ilamr , TCKMAX ) ;
TCD ( ilamg , TCKMAX ) ;
TCD ( n0_r , TCKMAX ) ;
TCD ( n0_g , TCKMAX ) ;
TCF ( mvd_r , TCKMAX ) ;
TCF ( mvd_c , TCKMAX ) ;
TCF ( smob , TCKMAX ) ;
TCF ( smo2 , TCKMAX ) ;
TCF ( smo1 , TCKMAX ) ;
TCF ( smo0 , TCKMAX ) ;
TCF ( smoc , TCKMAX ) ;
TCF ( smod , TCKMAX ) ;
TCF ( smoe , TCKMAX ) ;
TCF ( smof , TCKMAX ) ;
TCF ( sed_r , TCKMAX ) ;
TCF ( sed_s , TCKMAX ) ;
TCF ( sed_g , TCKMAX ) ;
TCF ( sed_i , TCKMAX ) ;
TCF ( sed_n , TCKMAX ) ;
TCF ( sed_c , TCKMAX ) ;
F rgvm ;
F delta_tp ;
F orho ;
F lfus2 ;
F onstep [ 5 ] ;
Soft8 n0_exp ;
Soft8 n0_min ;
Soft8 lam_exp ;
Soft8 lamc ;
Soft8 lamr ;
Soft8 lamg ;
Soft8 lami ;
Soft8 ilami ;
Soft8 ilamc ;
F xdc ;
F dc_b ;
F dc_g ;
F xdi ;
F xdr ;
F xds ;
F xdg ;
F ds_m ;
F dg_m ;
Soft8 dr_star ;
Soft8 dc_star ;
F zeta1 ;
F zeta ;
F taud ;
F tau ;
F stoke_r ;
F stoke_s ;
F stoke_g ;
F stoke_i ;
F vti ;
F vtr ;
F vts ;
F vtg ;
F vtc ;
TCF ( vtik , TCKMAX + 1 ) ;
TCF ( vtnik , TCKMAX + 1 ) ;
TCF ( vtrk , TCKMAX + 1 ) ;
TCF ( vtnrk , TCKMAX + 1 ) ;
TCF ( vtsk , TCKMAX + 1 ) ;
TCF ( vtgk , TCKMAX + 1 ) ;
TCF ( vtck , TCKMAX + 1 ) ;
TCF ( vtnck , TCKMAX + 1 ) ;
TCF ( vts_boost , TCKMAX ) ;
F mrat ;
F ils1 ;
F ils2 ;
F t1_vts ;
F t2_vts ;
F t3_vts ;
F t4_vts ;
F c_snow ;
F a_ ;
F b_ ;
F loga_ ;
F a1 ;
F a2 ;
F tf ;
F tempc ;
F tc0 ;
F r_mvd1 ;
F r_mvd2 ;
F xkrat ;
F xnc ;
F xri ;
F xni ;
F xmi ;
F oxmi ;
F xrc ;
F xrr ;
F xnr ;
F xsat ;
F rate_max ;
F sump ;
F ratio ;
F clap ;
F fcd ;
F dfcd ;
F otemp ;
F rvs ;
F rvs_p ;
F rvs_pp ;
F gamsc ;
F alphsc ;
F t1_evap ;
F t1_subl ;
F r_frac ;
F g_frac ;
F ef_rw ;
F ef_sw ;
F ef_gw ;
F ef_rr ;
F ef_ra ;
F ef_sa ;
F ef_ga ;
F dtsave ;
F odts ;
F odt ;
F odzq ;
F hgt_agl ;
F sr ;
F xslw1 ;
F ygra1 ;
F zans1 ;
F eva_factor ;
int i ;
int k ;
int k2 ;
int n ;
int nn ;
int nstep ;
int k_0 ;
int kbot ;
int it ;
int iexfrq ;
int ksed1 [ 5 ] ;
int nir ;
int nis ;
int nig ;
int nii ;
int nic ;
int niin ;
int idx_tc ;
int idx_t ;
int idx_s ;
int idx_g1 ;
int idx_g ;
int idx_r1 ;
int idx_r ;
int idx_i1 ;
int idx_i ;
int idx_c ;
int idx ;
int idx_d ;
int idx_n ;
int idx_in ;
bool melti ;
bool no_micro ;
TCB ( l_qc , TCKMAX ) ;
TCB ( l_qi , TCKMAX ) ;
TCB ( l_qr , TCKMAX ) ;
TCB ( l_qs , TCKMAX ) ;
TCB ( l_qg , TCKMAX ) ;
bool debug_flag ;
int mp_debug ;
int nu_c ;
TCB ( l_nifa , TCKMAX ) ;
TCB ( l_nwfa , TCKMAX ) ;
F tem ;
debug_flag = false ;
if ( debug_flag ) {
}
no_micro = true ;
dtsave = dt ;
odt = tc_div ( 1.f , dt ) ;
odts = tc_div ( 1.f , dtsave ) ;
iexfrq = 1 ;
for ( k = kts ;
k <= kte ;
k += 1 ) {
tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
qsten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
qgten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
nwfaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
nifaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prw_vcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnc_wcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnc_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnc_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnc_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnc_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnr_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnr_rcr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnr_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pri_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pni_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pri_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pni_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pni_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pri_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pni_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pni_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pri_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pni_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pni_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pni_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pri_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pni_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prs_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prs_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prs_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prg_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prg_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prg_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prg_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prg_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pna_rca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pna_sca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pna_gca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnd_rcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnd_scd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pnd_gcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
rainprod [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
evapprod [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
}
if ( is_aerosol_aware ) {
for ( k = kts ;
k <= kte ;
k += 1 ) {
l_nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
l_nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
smo0 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smo1 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smob [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smoc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smod [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smoe [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = t1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ;
qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 1.e-10f , qv1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = p1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ;
rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( 0.622f , pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_mul ( tc_mul ( r , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 0.622f ) ) ) ) ) ;
nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 11.1e6f , tc_min ( 9999.e6f , tc_mul ( nwfa1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( tc_mul ( nain1 , 0.01f ) , tc_min ( 9999.e6f , tc_mul ( nifa1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
if ( ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
no_micro = false ;
rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 2.f , tc_min ( tc_mul ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , nt_c_max ) ) ;
l_qc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
if ( ( is_aerosol_aware && ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= 2.f ) ) ) {
l_nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
nu_c = tc_min ( 15 , tc_add ( ( ( int ) round ( tc_div ( 1000.e6f , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , 2 ) ) ;
lamc = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , am_r ) , ccg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] ) , ocg1 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
xdc = tc_div ( ( tc_add ( tc_add ( bm_r , nu_c ) , 1.f ) ) , lamc ) ;
if ( ( xdc < d0c ) ) {
lamc = tc_div ( cce [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , d0c ) ;
}
else if ( ( xdc > tc_mul ( d0r , 2.f ) ) ) {
lamc = tc_div ( cce [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , ( tc_mul ( d0r , 2.f ) ) ) ;
}
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( nt_c_max ) ) , tc_mul ( tc_div ( tc_mul ( tc_mul ( ccg [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , ocg2 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , am_r ) , tc_pow ( lamc , bm_r ) ) ) ;
if ( ( ! is_aerosol_aware ) ) {
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = nt_c ;
}
}
else {
qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 2.f ;
l_qc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
no_micro = false ;
ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r2 , tc_mul ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
l_qi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
if ( ( is_aerosol_aware && ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r2 ) ) ) {
l_nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
if ( ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r2 ) ) {
lami = tc_div ( cie [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , 5.e-6f ) ;
ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( SD8(0x4163125300000000ul) , tc_mul ( tc_div ( tc_mul ( tc_mul ( cig [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , oig2 ) , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , am_i ) , tc_pow ( lami , bm_i ) ) ) ;
}
lami = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_i , cig [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , oig1 ) , ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmi ) ;
ilami = tc_div ( 1.f , lami ) ;
xdi = tc_mul ( ( tc_add ( tc_add ( bm_i , mu_i ) , 1.f ) ) , ilami ) ;
if ( ( xdi < 5.e-6f ) ) {
lami = tc_div ( cie [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , 5.e-6f ) ;
ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( SD8(0x4163125300000000ul) , tc_mul ( tc_div ( tc_mul ( tc_mul ( cig [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , oig2 ) , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , am_i ) , tc_pow ( lami , bm_i ) ) ) ;
}
else if ( ( xdi > 300.e-6f ) ) {
lami = tc_div ( cie [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , 300.e-6f ) ;
ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_div ( tc_mul ( tc_mul ( cig [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , oig2 ) , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , am_i ) , tc_pow ( lami , bm_i ) ) ;
}
}
else {
qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r2 ;
l_qi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
no_micro = false ;
rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r2 , tc_mul ( nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
if ( ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r2 ) ) {
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 1.0e-3f ;
lamr = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( tc_mul ( tc_mul ( crg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , org3 ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( lamr , bm_r ) ) , am_r ) ;
}
l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
lamr = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_r , crg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , org2 ) , nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , lamr ) ;
if ( ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 2.5e-3f ) ) {
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 2.5e-3f ;
lamr = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( tc_mul ( tc_mul ( crg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , org3 ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( lamr , bm_r ) ) , am_r ) ;
}
else if ( ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < tc_mul ( d0r , 0.75f ) ) ) {
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( d0r , 0.75f ) ;
lamr = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( tc_mul ( tc_mul ( crg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , org3 ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( lamr , bm_r ) ) , am_r ) ;
}
}
else {
qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r2 ;
l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
no_micro = false ;
rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
else {
qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
no_micro = false ;
rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
else {
qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
tempc = tc_sub ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) ;
rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( tc_div ( rho_not , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
rhof2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = rslf ( pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
delqvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 0.0f , tc_sub ( rslf ( pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) , qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
if ( ( tempc <= 0.0f ) ) {
qvsi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = rsif ( pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
else {
qvsi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ;
}
satw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
sati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvsi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( satw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 1.f ) ;
ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( sati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 1.f ) ;
if ( ( tc_abs ( ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) < eps ) ) {
ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
}
if ( ( tc_abs ( ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) < eps ) ) {
ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
}
if ( ( no_micro && ( ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 0.0f ) ) ) {
no_micro = false ;
}
diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( 2.11e-5f , tc_pow ( ( tc_div ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) ) , 1.94f ) ) , ( tc_div ( 101325.f , pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
if ( ( tempc >= 0.0f ) ) {
visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_add ( 1.718f , tc_mul ( 0.0049f , tempc ) ) ) , 1.0e-5f ) ;
}
else {
visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_sub ( tc_add ( 1.718f , tc_mul ( 0.0049f , tempc ) ) , tc_mul ( tc_mul ( 1.2e-5f , tempc ) , tempc ) ) ) , 1.0e-5f ) ;
}
ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( 1.f , ( tc_mul ( cp , ( tc_add ( 1.f , tc_mul ( 0.887f , qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ) ) ;
vsc2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( tc_div ( rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( lvap0 , tc_mul ( ( tc_sub ( 2106.0f , 4218.0f ) ) , tempc ) ) ;
tcond [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_add ( 5.69f , tc_mul ( 0.0168f , tempc ) ) ) , 1.0e-5f ) , 418.936f ) ;
}
if ( no_micro ) {
return ;
}
if ( ( ! iiwarm ) ) {
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
loga_ = tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) ;
smo0 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( a_ , tc_pow ( smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , b_ ) ) ;
loga_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , sa [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , tc_mul ( sa [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , sa [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( sa [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , sa [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , sb [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , tc_mul ( sb [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , sb [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( sb [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , sb [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] ) ;
smo1 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( a_ , tc_pow ( smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , b_ ) ) ;
loga_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sa [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sb [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ;
smoc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( a_ , tc_pow ( smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , b_ ) ) ;
loga_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sa [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sb [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 13 , ( 1 ) ) * ( 1 ) ] ) ) ;
smoe [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( a_ , tc_pow ( smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , b_ ) ) ;
loga_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sa [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sb [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 16 , ( 1 ) ) * ( 1 ) ] ) ) ;
smof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( a_ , tc_pow ( smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , b_ ) ) ;
}
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
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
lamr = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_r , crg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , org2 ) , nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( 1.f , lamr ) ;
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , lamr ) ;
n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , org2 ) , tc_pow ( lamr , cre [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) ;
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
if ( ( l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] && ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > d0r ) ) ) {
ef_rr = tc_sub ( 1.0f , tc_exp ( tc_mul ( 2300.0f , ( tc_sub ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 1950.0e-6f ) ) ) ) ) ;
pnr_rcr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( ef_rr , 2.0f ) , nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = d0c ;
if ( l_qc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) {
nu_c = tc_min ( 15 , tc_add ( ( ( int ) round ( tc_div ( 1000.e6f , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , 2 ) ) ;
xdc = tc_max ( tc_mul ( d0c , 1.e6f ) , tc_mul ( ( tc_pow ( ( tc_div ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( tc_mul ( am_r , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) , obmr ) ) , 1.e6f ) ) ;
lamc = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , am_r ) , ccg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] ) , ocg1 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( ( tc_add ( tc_add ( 3.0f , nu_c ) , 0.672f ) ) , lamc ) ;
}
if ( ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 0.01e-3f ) ) {
dc_g = tc_mul ( ( tc_div ( tc_pow ( ( tc_mul ( ccg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , ocg2 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) ) , obmr ) , lamc ) ) , 1.e6f ) ;
dc_b = tc_pow ( ( tc_sub ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( xdc , xdc ) , xdc ) , dc_g ) , dc_g ) , dc_g ) , tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( xdc , xdc ) , xdc ) , xdc ) , xdc ) , xdc ) ) ) , ( tc_div ( 1.f , 6.f ) ) ) ;
zeta1 = tc_mul ( 0.5f , ( tc_add ( ( tc_sub ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( 6.25e-6f , xdc ) , dc_b ) , dc_b ) , dc_b ) , 0.4f ) ) , tc_abs ( tc_sub ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( 6.25e-6f , xdc ) , dc_b ) , dc_b ) , dc_b ) , 0.4f ) ) ) ) ) ;
zeta = tc_mul ( tc_mul ( 0.027f , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , zeta1 ) ;
taud = tc_add ( tc_mul ( 0.5f , ( tc_add ( ( tc_sub ( tc_mul ( 0.5f , dc_b ) , 7.5f ) ) , tc_abs ( tc_sub ( tc_mul ( 0.5f , dc_b ) , 7.5f ) ) ) ) ) , r1 ) ;
tau = tc_div ( 3.72f , ( tc_mul ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , taud ) ) ) ;
prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( zeta , tau ) ;
prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pnr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( am_r , nu_c ) , 200.f ) , d0r ) , d0r ) , d0r ) ) ) ;
pnc_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , tc_div ( prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( tc_mul ( tc_mul ( tc_mul ( am_r , mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ;
}
if ( ( ( l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] && ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > d0r ) ) && ( mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > d0c ) ) ) {
lamr = tc_div ( 1.f , ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
idx = tc_add ( 1 , ( ( int ) ( tc_div ( tc_mul ( nbr , tc_log ( tc_div ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dr [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) , tc_log ( tc_div ( dr [ 0 + tc_sub ( nbr , ( 1 ) ) * ( 1 ) ] , dr [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ) ;
idx = tc_min ( idx , nbr ) ;
ef_rw = t_efrw [ 0 + tc_sub ( idx , ( 1 ) ) * ( 1 ) + tc_sub ( ( ( int ) ( tc_mul ( mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 1.e6f ) ) ) , ( 1 ) ) * tc_mul ( 1 , ( ( 100 ) - ( 1 ) + 1 ) ) ] ;
prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qr_qc ) , ef_rw ) , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_pow ( ( tc_add ( lamr , fv_r ) ) , ( ( - cre [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pnc_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qr_qc ) , ef_rw ) , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_pow ( ( tc_add ( lamr , fv_r ) ) , ( ( - cre [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
pnc_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pnc_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
if ( ( l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] && ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > d0r ) ) ) {
ef_ra = eff_aero ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 0.04e-6f , visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 114 ) ;
lamr = tc_div ( 1.f , ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pna_rca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qr_qc ) , ef_ra ) , nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_pow ( ( tc_add ( lamr , fv_r ) ) , ( ( - cre [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
pna_rca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pna_rca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ef_ra = eff_aero ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 0.8e-6f , visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 114 ) ;
pnd_rcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qr_qc ) , ef_ra ) , nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_pow ( ( tc_add ( lamr , fv_r ) ) , ( ( - cre [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
pnd_rcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pnd_rcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
}
if ( ( ! iiwarm ) ) {
for ( k = kts ;
k <= kte ;
k += 1 ) {
vts_boost [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 1.0f ;
xds = 0.0f ;
if ( l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) {
xds = tc_div ( smoc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , smob [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
tempc = tc_sub ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) ;
idx_tc = tc_max ( 1 , tc_min ( ( ( int ) round ( ( - tempc ) ) ) , 45 ) ) ;
idx_t = tc_sub ( ( ( int ) ( tc_div ( ( tc_sub ( tempc , 2.5f ) ) , 5.f ) ) ) , 1 ) ;
idx_t = tc_max ( 1 , ( - idx_t ) ) ;
idx_t = tc_min ( idx_t , ntb_t ) ;
it = tc_max ( 1 , tc_min ( ( ( int ) round ( ( - tempc ) ) ) , 31 ) ) ;
if ( ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r_c [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
nic = ( ( int ) round ( tc_log10 ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
for ( nn = tc_sub ( nic , 1 ) ;
nn <= tc_add ( nic , 1 ) ;
nn += 1 ) {
n = nn ;
if ( ( ( ( tc_div ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) >= 1.0f ) && ( ( tc_div ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) < 10.0f ) ) ) {
break ;
}
}
idx_c = tc_sub ( tc_add ( ( ( int ) ( tc_div ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , n ) ) ) ) , tc_mul ( 10 , ( tc_sub ( n , nic2 ) ) ) ) , ( tc_sub ( n , nic2 ) ) ) ;
idx_c = tc_max ( 1 , tc_min ( idx_c , ntb_c ) ) ;
}
else {
idx_c = 1 ;
}
idx_n = ( ( int ) round ( tc_add ( 1.0f , tc_div ( tc_mul ( F ( nbc ) , tc_log ( tc_div ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t_nc [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) , nic1 ) ) ) ) ;
idx_n = tc_max ( 1 , tc_min ( idx_n , nbc ) ) ;
if ( ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r_i [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
nii = ( ( int ) round ( tc_log10 ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
for ( nn = tc_sub ( nii , 1 ) ;
nn <= tc_add ( nii , 1 ) ;
nn += 1 ) {
n = nn ;
if ( ( ( ( tc_div ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) >= 1.0f ) && ( ( tc_div ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) < 10.0f ) ) ) {
break ;
}
}
idx_i = tc_sub ( tc_add ( ( ( int ) ( tc_div ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , n ) ) ) ) , tc_mul ( 10 , ( tc_sub ( n , nii2 ) ) ) ) , ( tc_sub ( n , nii2 ) ) ) ;
idx_i = tc_max ( 1 , tc_min ( idx_i , ntb_i ) ) ;
}
else {
idx_i = 1 ;
}
if ( ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > nt_i [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
nii = ( ( int ) round ( tc_log10 ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
for ( nn = tc_sub ( nii , 1 ) ;
nn <= tc_add ( nii , 1 ) ;
nn += 1 ) {
n = nn ;
if ( ( ( ( tc_div ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) >= 1.0f ) && ( ( tc_div ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) < 10.0f ) ) ) {
break ;
}
}
idx_i1 = tc_sub ( tc_add ( ( ( int ) ( tc_div ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , n ) ) ) ) , tc_mul ( 10 , ( tc_sub ( n , nii3 ) ) ) ) , ( tc_sub ( n , nii3 ) ) ) ;
idx_i1 = tc_max ( 1 , tc_min ( idx_i1 , ntb_i1 ) ) ;
}
else {
idx_i1 = 1 ;
}
if ( ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r_r [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
nir = ( ( int ) round ( tc_log10 ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
for ( nn = tc_sub ( nir , 1 ) ;
nn <= tc_add ( nir , 1 ) ;
nn += 1 ) {
n = nn ;
if ( ( ( ( tc_div ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) >= 1.0f ) && ( ( tc_div ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) < 10.0f ) ) ) {
break ;
}
}
idx_r = tc_sub ( tc_add ( ( ( int ) ( tc_div ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , n ) ) ) ) , tc_mul ( 10 , ( tc_sub ( n , nir2 ) ) ) ) , ( tc_sub ( n , nir2 ) ) ) ;
idx_r = tc_max ( 1 , tc_min ( idx_r , ntb_r ) ) ;
lamr = tc_div ( 1.f , ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
lam_exp = tc_mul ( lamr , tc_pow ( ( tc_mul ( tc_mul ( crg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , org2 ) , org1 ) ) , bm_r ) ) ;
n0_exp = tc_mul ( tc_div ( tc_mul ( org1 , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , am_r ) , tc_pow ( lam_exp , cre [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ;
nir = ( ( int ) round ( tc_log10 ( n0_exp ) ) ) ;
for ( nn = tc_sub ( nir , 1 ) ;
nn <= tc_add ( nir , 1 ) ;
nn += 1 ) {
n = nn ;
if ( ( ( ( tc_div ( n0_exp , tc_pow ( 10.f , nn ) ) ) >= 1.0f ) && ( ( tc_div ( n0_exp , tc_pow ( 10.f , nn ) ) ) < 10.0f ) ) ) {
break ;
}
}
idx_r1 = tc_sub ( tc_add ( ( ( int ) ( tc_div ( n0_exp , tc_pow ( 10.f , n ) ) ) ) , tc_mul ( 10 , ( tc_sub ( n , nir3 ) ) ) ) , ( tc_sub ( n , nir3 ) ) ) ;
idx_r1 = tc_max ( 1 , tc_min ( idx_r1 , ntb_r1 ) ) ;
}
else {
idx_r = 1 ;
idx_r1 = ntb_r1 ;
}
if ( ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r_s [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
nis = ( ( int ) round ( tc_log10 ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
for ( nn = tc_sub ( nis , 1 ) ;
nn <= tc_add ( nis , 1 ) ;
nn += 1 ) {
n = nn ;
if ( ( ( ( tc_div ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) >= 1.0f ) && ( ( tc_div ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) < 10.0f ) ) ) {
break ;
}
}
idx_s = tc_sub ( tc_add ( ( ( int ) ( tc_div ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , n ) ) ) ) , tc_mul ( 10 , ( tc_sub ( n , nis2 ) ) ) ) , ( tc_sub ( n , nis2 ) ) ) ;
idx_s = tc_max ( 1 , tc_min ( idx_s , ntb_s ) ) ;
}
else {
idx_s = 1 ;
}
if ( ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r_g [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
nig = ( ( int ) round ( tc_log10 ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
for ( nn = tc_sub ( nig , 1 ) ;
nn <= tc_add ( nig , 1 ) ;
nn += 1 ) {
n = nn ;
if ( ( ( ( tc_div ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) >= 1.0f ) && ( ( tc_div ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) < 10.0f ) ) ) {
break ;
}
}
idx_g = tc_sub ( tc_add ( ( ( int ) ( tc_div ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , n ) ) ) ) , tc_mul ( 10 , ( tc_sub ( n , nig2 ) ) ) ) , ( tc_sub ( n , nig2 ) ) ) ;
idx_g = tc_max ( 1 , tc_min ( idx_g , ntb_g ) ) ;
lamg = tc_div ( 1.f , ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
lam_exp = tc_mul ( lamg , tc_pow ( ( tc_mul ( tc_mul ( cgg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , ogg2 ) , ogg1 ) ) , bm_g ) ) ;
n0_exp = tc_mul ( tc_div ( tc_mul ( ogg1 , rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , am_g ) , tc_pow ( lam_exp , cge [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ;
nig = ( ( int ) round ( tc_log10 ( n0_exp ) ) ) ;
for ( nn = tc_sub ( nig , 1 ) ;
nn <= tc_add ( nig , 1 ) ;
nn += 1 ) {
n = nn ;
if ( ( ( ( tc_div ( n0_exp , tc_pow ( 10.f , nn ) ) ) >= 1.0f ) && ( ( tc_div ( n0_exp , tc_pow ( 10.f , nn ) ) ) < 10.0f ) ) ) {
break ;
}
}
idx_g1 = tc_sub ( tc_add ( ( ( int ) ( tc_div ( n0_exp , tc_pow ( 10.f , n ) ) ) ) , tc_mul ( 10 , ( tc_sub ( n , nig3 ) ) ) ) , ( tc_sub ( n , nig3 ) ) ) ;
idx_g1 = tc_max ( 1 , tc_min ( idx_g1 , ntb_g1 ) ) ;
}
else {
idx_g = 1 ;
idx_g1 = ntb_g1 ;
}
otemp = tc_div ( 1.f , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rvs = tc_mul ( rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvsi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rvs_p = tc_mul ( tc_mul ( rvs , otemp ) , ( tc_sub ( tc_mul ( tc_mul ( lsub , otemp ) , orv ) , 1.f ) ) ) ;
rvs_pp = tc_mul ( rvs , ( tc_add ( tc_add ( tc_mul ( tc_mul ( tc_mul ( otemp , ( tc_sub ( tc_mul ( tc_mul ( lsub , otemp ) , orv ) , 1.f ) ) ) , otemp ) , ( tc_sub ( tc_mul ( tc_mul ( lsub , otemp ) , orv ) , 1.f ) ) ) , ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( ( - 2.f ) , lsub ) , otemp ) , otemp ) , otemp ) , orv ) ) ) , tc_mul ( otemp , otemp ) ) ) ) ;
gamsc = tc_mul ( tc_div ( tc_mul ( lsub , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tcond [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rvs_p ) ;
alphsc = tc_div ( tc_mul ( tc_div ( tc_mul ( tc_mul ( tc_mul ( 0.5f , ( tc_div ( gamsc , ( tc_add ( 1.f , gamsc ) ) ) ) ) , ( tc_div ( gamsc , ( tc_add ( 1.f , gamsc ) ) ) ) ) , rvs_pp ) , rvs_p ) , rvs ) , rvs_p ) ;
alphsc = tc_max ( 1.e-9f , alphsc ) ;
xsat = ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ;
if ( ( tc_abs ( xsat ) < 1.e-9f ) ) {
xsat = 0.f ;
}
t1_subl = tc_div ( tc_mul ( tc_mul ( 4.f , pi ) , ( tc_sub ( tc_add ( tc_sub ( 1.0f , tc_mul ( alphsc , xsat ) ) , tc_mul ( tc_mul ( tc_mul ( tc_mul ( 2.f , alphsc ) , alphsc ) , xsat ) , xsat ) ) , tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( 5.f , alphsc ) , alphsc ) , alphsc ) , xsat ) , xsat ) , xsat ) ) ) ) , ( tc_add ( 1.f , gamsc ) ) ) ;
if ( ( l_qc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] && ( mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > d0c ) ) ) {
if ( ( xds > d0s ) ) {
idx = tc_add ( 1 , ( ( int ) ( tc_div ( tc_mul ( nbs , tc_log ( tc_div ( xds , ds [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) , tc_log ( tc_div ( ds [ 0 + tc_sub ( nbs , ( 1 ) ) * ( 1 ) ] , ds [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ) ;
idx = tc_min ( idx , nbs ) ;
ef_sw = t_efsw [ 0 + tc_sub ( idx , ( 1 ) ) * ( 1 ) + tc_sub ( ( ( int ) ( tc_mul ( mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 1.e6f ) ) ) , ( 1 ) ) * tc_mul ( 1 , ( ( 100 ) - ( 1 ) + 1 ) ) ] ;
prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qs_qc ) , ef_sw ) , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , smoe [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pnc_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qs_qc ) , ef_sw ) , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , smoe [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pnc_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pnc_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
if ( ( ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] >= r_g [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) && ( mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > d0c ) ) ) {
xdg = tc_mul ( ( tc_add ( tc_add ( bm_g , mu_g ) , 1.f ) ) , ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
vtg = tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , av_g ) , cgg [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] ) , ogg3 ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , bv_g ) ) ;
stoke_g = tc_div ( tc_mul ( tc_mul ( tc_mul ( mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , mvd_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , vtg ) , rho_w ) , ( tc_mul ( tc_mul ( 9.f , visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , xdg ) ) ) ;
if ( ( xdg > d0g ) ) {
if ( ( ( stoke_g >= 0.4f ) && ( stoke_g <= 10.f ) ) ) {
ef_gw = tc_mul ( 0.55f , tc_log10 ( tc_mul ( 2.51f , stoke_g ) ) ) ;
}
else if ( ( stoke_g < 0.4f ) ) {
ef_gw = 0.0f ;
}
else if ( ( stoke_g > 10 ) ) {
ef_gw = 0.77f ;
}
prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qg_qc ) , ef_gw ) , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] ) ) ;
pnc_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qg_qc ) , ef_gw ) , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] ) ) ;
pnc_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pnc_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
}
}
if ( ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r_s [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
ef_sa = eff_aero ( xds , 0.04e-6f , visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 115 ) ;
pna_sca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qs_qc ) , ef_sa ) , nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , smoe [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pna_sca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pna_sca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ef_sa = eff_aero ( xds , 0.8e-6f , visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 115 ) ;
pnd_scd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qs_qc ) , ef_sa ) , nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , smoe [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pnd_scd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pnd_scd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
if ( ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r_g [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
xdg = tc_mul ( ( tc_add ( tc_add ( bm_g , mu_g ) , 1.f ) ) , ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ef_ga = eff_aero ( xdg , 0.04e-6f , visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 103 ) ;
pna_gca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qg_qc ) , ef_ga ) , nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] ) ) ;
pna_gca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pna_gca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ef_ga = eff_aero ( xdg , 0.8e-6f , visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 103 ) ;
pnd_gcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qg_qc ) , ef_ga ) , nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] ) ) ;
pnd_gcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pnd_gcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
if ( ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] >= r_r [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
if ( ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] >= r_s [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
if ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < t_0 ) ) {
prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = ( - ( tc_add ( tc_add ( tc_add ( tmr_racs2 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] , tcr_sacr2 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tmr_racs1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tcr_sacr1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) ) ) ;
prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( tc_sub ( tc_add ( tmr_racs2 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] , tcr_sacr2 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tcs_racs1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tms_sacr1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) ;
prg_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tc_add ( tc_add ( tmr_racs1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] , tcr_sacr1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tcs_racs1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tms_sacr1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) ;
prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( ( - rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ) ) , prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( ( - rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ) ) , prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prg_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( ( tc_add ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odts ) ) ) , prg_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pnr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tc_add ( tc_add ( tnr_racs1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] , tnr_racs2 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tnr_sacr1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tnr_sacr2 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) ;
pnr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pnr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
else {
prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tc_add ( tc_sub ( ( - tcs_racs1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tms_sacr1 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tmr_racs2 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , tcr_sacr2 [ 0 + tc_sub ( idx_s , ( 1 ) ) * ( 1 ) + tc_sub ( idx_t , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 9 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) ;
prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( ( - rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ) ) , prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = ( - prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
}
if ( ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] >= r_g [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
if ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < t_0 ) ) {
prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tmr_racg [ 0 + tc_sub ( idx_g1 , ( 1 ) ) * ( 1 ) + tc_sub ( idx_g , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] , tcr_gacr [ 0 + tc_sub ( idx_g1 , ( 1 ) ) * ( 1 ) + tc_sub ( idx_g , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) ;
prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = ( - prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pnr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tnr_racg [ 0 + tc_sub ( idx_g1 , ( 1 ) ) * ( 1 ) + tc_sub ( idx_g , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] , tnr_gacr [ 0 + tc_sub ( idx_g1 , ( 1 ) ) * ( 1 ) + tc_sub ( idx_g , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) ;
pnr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pnr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
else {
prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tcg_racg [ 0 + tc_sub ( idx_g1 , ( 1 ) ) * ( 1 ) + tc_sub ( idx_g , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ;
prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = ( - prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pnr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( - 5.0f ) , tnr_gacr [ 0 + tc_sub ( idx_g1 , ( 1 ) ) * ( 1 ) + tc_sub ( idx_g , ( 1 ) ) * tc_mul ( 1 , ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r1 , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_r , ( 1 ) ) * ( 1 * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 28 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) ;
}
}
}
if ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < t_0 ) ) {
vts_boost [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 1.0f ;
rate_max = tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvsi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) , 0.999f ) ;
if ( ( dustyice && is_aerosol_aware ) ) {
xni = icedemott ( tempc , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvsi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
else {
xni = tc_mul ( 1.0f , 1000.f ) ;
}
if ( ( xni > nt_in [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
niin = ( ( int ) round ( tc_log10 ( xni ) ) ) ;
for ( nn = tc_sub ( niin , 1 ) ;
nn <= tc_add ( niin , 1 ) ;
nn += 1 ) {
n = nn ;
if ( ( ( ( tc_div ( xni , tc_pow ( 10.f , nn ) ) ) >= 1.0f ) && ( ( tc_div ( xni , tc_pow ( 10.f , nn ) ) ) < 10.0f ) ) ) {
break ;
}
}
idx_in = tc_sub ( tc_add ( ( ( int ) ( tc_div ( xni , tc_pow ( 10.f , n ) ) ) ) , tc_mul ( 10 , ( tc_sub ( n , niin2 ) ) ) ) , ( tc_sub ( n , niin2 ) ) ) ;
idx_in = tc_max ( 1 , tc_min ( idx_in , ntb_in ) ) ;
}
else {
idx_in = 1 ;
}
if ( ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r_r [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
prg_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tpg_qrfz [ 0 + tc_sub ( idx_r , ( 1 ) ) * ( 1 ) + tc_sub ( idx_r1 , ( 1 ) ) * tc_mul ( 1 , ( ( 37 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_tc , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_in , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 45 ) - ( 1 ) + 1 ) ) ] , odts ) ;
pri_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tpi_qrfz [ 0 + tc_sub ( idx_r , ( 1 ) ) * ( 1 ) + tc_sub ( idx_r1 , ( 1 ) ) * tc_mul ( 1 , ( ( 37 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_tc , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_in , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 45 ) - ( 1 ) + 1 ) ) ] , odts ) ;
pni_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tni_qrfz [ 0 + tc_sub ( idx_r , ( 1 ) ) * ( 1 ) + tc_sub ( idx_r1 , ( 1 ) ) * tc_mul ( 1 , ( ( 37 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_tc , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_in , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 45 ) - ( 1 ) + 1 ) ) ] , odts ) ;
pnr_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tnr_qrfz [ 0 + tc_sub ( idx_r , ( 1 ) ) * ( 1 ) + tc_sub ( idx_r1 , ( 1 ) ) * tc_mul ( 1 , ( ( 37 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_tc , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_in , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 45 ) - ( 1 ) + 1 ) ) ] , odts ) ;
pnr_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pnr_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
else if ( ( ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) && ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < hgfr ) ) ) {
pri_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ;
pni_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ;
}
if ( ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r_c [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tpi_qcfz [ 0 + tc_sub ( idx_c , ( 1 ) ) * ( 1 ) + tc_sub ( idx_n , ( 1 ) ) * tc_mul ( 1 , ( ( 37 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_tc , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 100 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_in , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 100 ) - ( 1 ) + 1 ) * ( ( 45 ) - ( 1 ) + 1 ) ) ] , odts ) ;
pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pni_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tni_qcfz [ 0 + tc_sub ( idx_c , ( 1 ) ) * ( 1 ) + tc_sub ( idx_n , ( 1 ) ) * tc_mul ( 1 , ( ( 37 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_tc , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 100 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_in , ( 1 ) ) * ( 1 * ( ( 37 ) - ( 1 ) + 1 ) * ( ( 100 ) - ( 1 ) + 1 ) * ( ( 45 ) - ( 1 ) + 1 ) ) ] , odts ) ;
pni_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , tc_div ( pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( tc_mul ( 2.f , xm0i ) ) ) , pni_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
else if ( ( ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) && ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < hgfr ) ) ) {
pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ;
pni_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ;
}
if ( ( ( ( ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] >= 0.25f ) ) || ( ( ( ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > eps ) && ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < 253.15f ) ) ) ) ) {
if ( ( dustyice && is_aerosol_aware ) ) {
xnc = icedemott ( tempc , qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvsi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , nifa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
else {
xnc = tc_min ( 250.e3f , tc_mul ( tno , tc_exp ( tc_mul ( ato , ( tc_sub ( t_0 , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ) ) ;
}
xni = tc_add ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_add ( pni_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pni_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , dtsave ) ) ;
pni_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( 0.5f , ( tc_add ( tc_sub ( xnc , xni ) , tc_abs ( tc_sub ( xnc , xni ) ) ) ) ) , odts ) ;
pri_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( rate_max ) ) , tc_mul ( xm0i , pni_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
pni_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( pri_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , xm0i ) ;
}
xni = tc_add ( tc_add ( smo0 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_mul ( ( tc_add ( tc_add ( pni_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pni_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , dtsave ) ) ;
if ( ( ( ( ( is_aerosol_aware && homogice ) && ( ( xni <= 999.e3f ) ) ) && ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < 238 ) ) ) && ( ( ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] >= 0.4f ) ) ) ) {
xnc = icekoop ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dtsave ) ;
pni_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( xnc , odts ) ;
pri_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( rate_max ) ) , tc_mul ( tc_mul ( xm0i , 0.1f ) , pni_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
pni_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( pri_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( tc_mul ( xm0i , 0.1f ) ) ) ;
}
if ( l_qi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) {
lami = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_i , cig [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , oig1 ) , ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmi ) ;
ilami = tc_div ( 1.f , lami ) ;
xdi = tc_max ( ( ( Soft8 ) ( d0i ) ) , tc_mul ( ( tc_add ( tc_add ( bm_i , mu_i ) , 1.f ) ) , ilami ) ) ;
xmi = tc_mul ( am_i , tc_pow ( xdi , bm_i ) ) ;
oxmi = tc_div ( 1.f , xmi ) ;
pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( c_cube , t1_subl ) , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rvs ) , oig1 ) , cig [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] ) , ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ilami ) ;
if ( ( pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < 0.0f ) ) {
pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( ( - ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ) ) , pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( ( Soft8 ) ( rate_max ) ) ) ;
pni_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , oxmi ) ;
pni_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( ( - ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ) ) , pni_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
else {
pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( ( Soft8 ) ( rate_max ) ) ) ;
prs_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_sub ( SD8(0x3ff0000000000000ul) , tpi_ide [ 0 + tc_sub ( idx_i , ( 1 ) ) * ( 1 ) + tc_sub ( idx_i1 , ( 1 ) ) * tc_mul ( 1 , ( ( 64 ) - ( 1 ) + 1 ) ) ] ) ) , pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tpi_ide [ 0 + tc_sub ( idx_i , ( 1 ) ) * ( 1 ) + tc_sub ( idx_i1 , ( 1 ) ) * tc_mul ( 1 , ( ( 64 ) - ( 1 ) + 1 ) ) ] , pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
if ( ( ( ( idx_i == ntb_i ) ) || ( ( xdi > tc_mul ( 5.0f , d0s ) ) ) ) ) {
prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , .99f ) , odts ) ;
pni_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , .95f ) , odts ) ;
}
else if ( ( xdi < tc_mul ( 0.1f , d0s ) ) ) {
prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
pni_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
}
else {
prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tps_iaus [ 0 + tc_sub ( idx_i , ( 1 ) ) * ( 1 ) + tc_sub ( idx_i1 , ( 1 ) ) * tc_mul ( 1 , ( ( 64 ) - ( 1 ) + 1 ) ) ] , odts ) ;
prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( tc_mul ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , .99f ) , odts ) ) ) , prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pni_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tni_iaus [ 0 + tc_sub ( idx_i , ( 1 ) ) * ( 1 ) + tc_sub ( idx_i1 , ( 1 ) ) * tc_mul ( 1 , ( ( 64 ) - ( 1 ) + 1 ) ) ] , odts ) ;
pni_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( tc_mul ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , .95f ) , odts ) ) ) , pni_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
}
if ( l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) {
c_snow = tc_add ( c_sqrd , tc_div ( tc_mul ( ( tc_add ( tempc , 1.5f ) ) , ( tc_sub ( c_cube , c_sqrd ) ) ) , ( tc_add ( ( - 30.f ) , 1.5f ) ) ) ) ;
c_snow = tc_max ( c_sqrd , tc_min ( c_snow , c_cube ) ) ;
prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( c_snow , t1_subl ) , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rvs ) , ( tc_add ( tc_mul ( t1_qs_sd , smo1 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_mul ( tc_mul ( tc_mul ( t2_qs_sd , rhof2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , vsc2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , smof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ;
if ( ( prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < 0.f ) ) {
prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( ( - rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ) ) , prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( ( Soft8 ) ( rate_max ) ) ) ;
}
else {
prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( ( Soft8 ) ( rate_max ) ) ) ;
}
}
if ( ( l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] && ( ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < ( - eps ) ) ) ) {
prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( c_cube , t1_subl ) , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rvs ) , n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( tc_mul ( t1_qg_sd , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( t2_qg_sd , vsc2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rhof2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 11 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
if ( ( prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < 0.f ) ) {
prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( ( - rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ) ) , prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( ( Soft8 ) ( rate_max ) ) ) ;
}
else {
prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( ( Soft8 ) ( rate_max ) ) ) ;
}
}
if ( l_qi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) {
lami = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_i , cig [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , oig1 ) , ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmi ) ;
ilami = tc_div ( 1.f , lami ) ;
xdi = tc_max ( ( ( Soft8 ) ( d0i ) ) , tc_mul ( ( tc_add ( tc_add ( bm_i , mu_i ) , 1.f ) ) , ilami ) ) ;
xmi = tc_mul ( am_i , tc_pow ( xdi , bm_i ) ) ;
oxmi = tc_div ( 1.f , xmi ) ;
if ( ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] >= r_s [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
prs_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( t1_qs_qi , rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ef_si ) , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , smoe [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pni_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prs_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , oxmi ) ;
}
if ( ( ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] >= r_r [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) && ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > tc_mul ( 4.f , xdi ) ) ) ) {
lamr = tc_div ( 1.f , ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pri_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qr_qi ) , ef_ri ) , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_pow ( ( tc_add ( lamr , fv_r ) ) , ( ( - cre [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
pnr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t1_qr_qi ) , ef_ri ) , ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_pow ( ( tc_add ( lamr , fv_r ) ) , ( ( - cre [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
pni_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( pri_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , oxmi ) ;
prr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t2_qr_qi ) , ef_ri ) , ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_pow ( ( tc_add ( lamr , fv_r ) ) , ( ( - cre [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
prr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , prr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prg_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( pri_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
}
if ( ( ( prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > eps ) && ( tempc > ( - 8.0f ) ) ) ) {
tf = 0.f ;
if ( ( ( tempc >= ( - 5.0f ) ) && ( tempc < ( - 3.0f ) ) ) ) {
tf = tc_mul ( 0.5f , ( tc_sub ( ( - 3.0f ) , tempc ) ) ) ;
}
else if ( ( ( tempc > ( - 8.0f ) ) && ( tempc < ( - 5.0f ) ) ) ) {
tf = tc_mul ( 0.33333333f , ( tc_add ( 8.0f , tempc ) ) ) ;
}
pni_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( 3.5e8f , tf ) , prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pri_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( xm0i , pni_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prs_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_div ( prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( tc_add ( prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , pri_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prg_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_div ( prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( tc_add ( prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , pri_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
if ( ( ( prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > tc_mul ( 2.0f , prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) && ( prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > eps ) ) ) {
r_frac = tc_min ( SD8(0x403e000000000000ul) , tc_div ( prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
g_frac = tc_min ( 0.95f , tc_add ( 0.15f , tc_mul ( ( tc_sub ( r_frac , 2.f ) ) , .028f ) ) ) ;
vts_boost [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( 1.5f , tc_add ( 1.1f , tc_mul ( ( tc_sub ( r_frac , 2.f ) ) , .014f ) ) ) ;
prg_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( g_frac , prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_sub ( 1.f , g_frac ) ) , prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
}
else {
if ( l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) {
prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_sub ( tc_mul ( tempc , tcond [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_mul ( tc_mul ( lvap0 , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , delqvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , ( tc_add ( tc_mul ( t1_qs_me , smo1 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_mul ( tc_mul ( tc_mul ( t2_qs_me , rhof2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , vsc2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , smof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ;
if ( ( prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 0.f ) ) {
prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( 4218.f , olfus ) , tempc ) , ( tc_add ( prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ;
}
prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , tc_max ( SD8(0x0000000000000000ul) , prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
pnr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_div ( smo0 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( 10.0f , ( tc_mul ( ( - 0.25f ) , tempc ) ) ) ) ;
pnr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( smo0 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , pnr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
if ( ( ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < 0.f ) ) {
prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( c_cube , t1_subl ) , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rvs ) , ( tc_add ( tc_mul ( t1_qs_sd , smo1 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_mul ( tc_mul ( tc_mul ( t2_qs_sd , rhof2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , vsc2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , smof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ;
prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( ( - rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ) ) , prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
}
if ( l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) {
prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_sub ( tc_mul ( tempc , tcond [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_mul ( tc_mul ( lvap0 , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , delqvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( tc_mul ( t1_qg_me , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( t2_qg_me , rhof2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , vsc2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 11 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odts ) ) ) , tc_max ( SD8(0x0000000000000000ul) , prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
pnr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_div ( tc_mul ( tc_mul ( n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cgg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) , rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( 10.0f , ( tc_mul ( ( - 0.5f ) , tempc ) ) ) ) ;
if ( ( ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < 0.f ) ) {
prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( c_cube , t1_subl ) , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ssati [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rvs ) , n0_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( tc_mul ( t1_qg_sd , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( t2_qg_sd , vsc2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rhof2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cge [ 0 + tc_sub ( 11 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( ( - rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ) ) , prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
}
if ( ( dt > 120.f ) ) {
prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tc_add ( prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
}
}
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
sump = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( pri_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rate_max = tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvsi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) , 0.999f ) ;
if ( ( ( ( ( sump > eps ) && ( sump > rate_max ) ) ) || ( ( ( sump < ( - eps ) ) && ( sump < rate_max ) ) ) ) ) {
ratio = tc_div ( rate_max , sump ) ;
pri_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( pri_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
pni_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( pni_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prs_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prs_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
pri_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( pri_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
}
sump = tc_sub ( tc_sub ( tc_sub ( tc_sub ( tc_sub ( ( - prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rate_max = tc_mul ( ( - rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ;
if ( ( ( sump < rate_max ) && l_qc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) {
ratio = tc_div ( rate_max , sump ) ;
prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prg_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prg_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
}
sump = tc_sub ( tc_sub ( tc_sub ( pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rate_max = tc_mul ( ( - ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ;
if ( ( ( sump < rate_max ) && l_qi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) {
ratio = tc_div ( rate_max , sump ) ;
pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prs_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prs_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
pri_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( pri_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
}
sump = tc_add ( tc_add ( tc_sub ( tc_sub ( ( - prg_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rate_max = tc_mul ( ( - rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ;
if ( ( ( sump < rate_max ) && l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) {
ratio = tc_div ( rate_max , sump ) ;
prg_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prg_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
pri_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( pri_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
}
sump = tc_add ( tc_sub ( tc_sub ( prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prs_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rate_max = tc_mul ( ( - rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ;
if ( ( ( sump < rate_max ) && l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) {
ratio = tc_div ( rate_max , sump ) ;
prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prs_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prs_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
}
sump = tc_add ( tc_sub ( tc_sub ( prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prg_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rate_max = tc_mul ( ( - rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ;
if ( ( ( sump < rate_max ) && l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) {
ratio = tc_div ( rate_max , sump ) ;
prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prg_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prg_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ratio ) ;
}
pri_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( prs_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prg_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ratio = tc_min ( tc_abs ( prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_abs ( prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ratio , tc_sign ( 1.0f , tc_sngl ( prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = ( - prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
if ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > t_0 ) ) {
ratio = tc_min ( tc_abs ( prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_abs ( prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ratio , tc_sign ( 1.0f , tc_sngl ( prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = ( - prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
lfus2 = tc_sub ( lsub , lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
if ( is_aerosol_aware ) {
nwfaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( nwfaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_add ( tc_add ( tc_add ( pna_rca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pna_sca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pna_gca [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , orho ) ) ;
nifaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( nifaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_add ( tc_add ( pnd_rcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pnd_scd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pnd_gcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , orho ) ) ;
if ( dustyice ) {
nifaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( nifaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( pni_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , orho ) ) ;
}
else {
nifaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
}
}
qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_sub ( tc_sub ( tc_sub ( tc_sub ( tc_sub ( ( - pri_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , orho ) ) ;
qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_sub ( tc_sub ( tc_sub ( tc_sub ( tc_sub ( ( - prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , orho ) ) ;
ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_sub ( tc_sub ( tc_sub ( tc_sub ( ( - pnc_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pnc_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pnc_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pnc_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , orho ) ) ;
xrc = tc_max ( r1 , tc_mul ( ( tc_add ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dtsave ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
xnc = tc_max ( 2.f , tc_mul ( ( tc_add ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dtsave ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
if ( ( xrc > r1 ) ) {
nu_c = tc_min ( 15 , tc_add ( ( ( int ) round ( tc_div ( 1000.e6f , xnc ) ) ) , 2 ) ) ;
lamc = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( xnc , am_r ) , ccg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] ) , ocg1 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
xdc = tc_div ( ( tc_add ( tc_add ( bm_r , nu_c ) , 1.f ) ) , lamc ) ;
if ( ( xdc < d0c ) ) {
lamc = tc_div ( cce [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , d0c ) ;
xnc = tc_mul ( tc_div ( tc_mul ( tc_mul ( ccg [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , ocg2 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , xrc ) , am_r ) , tc_pow ( lamc , bm_r ) ) ;
ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_sub ( xnc , tc_mul ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , odts ) , orho ) ;
}
else if ( ( xdc > tc_mul ( d0r , 2.f ) ) ) {
lamc = tc_div ( cce [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , ( tc_mul ( d0r , 2.f ) ) ) ;
xnc = tc_mul ( tc_div ( tc_mul ( tc_mul ( ccg [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , ocg2 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , xrc ) , am_r ) , tc_pow ( lamc , bm_r ) ) ;
ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_sub ( xnc , tc_mul ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , odts ) , orho ) ;
}
}
else {
ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( - nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ;
}
xnc = tc_max ( 0.f , tc_mul ( ( tc_add ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dtsave ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
if ( ( xnc > nt_c_max ) ) {
ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_sub ( nt_c_max , tc_mul ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , odts ) , orho ) ;
}
qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_sub ( tc_sub ( tc_sub ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( pri_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pri_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , orho ) ) ;
niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_sub ( tc_sub ( tc_sub ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( pni_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pni_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , orho ) ) ;
xri = tc_max ( r1 , tc_mul ( ( tc_add ( qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dtsave ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
xni = tc_max ( r2 , tc_mul ( ( tc_add ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dtsave ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
if ( ( xri > r1 ) ) {
lami = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_i , cig [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , oig1 ) , xni ) , xri ) ) , obmi ) ;
ilami = tc_div ( 1.f , lami ) ;
xdi = tc_mul ( ( tc_add ( tc_add ( bm_i , mu_i ) , 1.f ) ) , ilami ) ;
if ( ( xdi < 5.e-6f ) ) {
lami = tc_div ( cie [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , 5.e-6f ) ;
xni = tc_min ( SD8(0x4163125300000000ul) , tc_mul ( tc_div ( tc_mul ( tc_mul ( cig [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , oig2 ) , xri ) , am_i ) , tc_pow ( lami , bm_i ) ) ) ;
niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_sub ( xni , tc_mul ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , odts ) , orho ) ;
}
else if ( ( xdi > 300.e-6f ) ) {
lami = tc_div ( cie [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , 300.e-6f ) ;
xni = tc_mul ( tc_div ( tc_mul ( tc_mul ( cig [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , oig2 ) , xri ) , am_i ) , tc_pow ( lami , bm_i ) ) ;
niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_sub ( xni , tc_mul ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , odts ) , orho ) ;
}
}
else {
niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( - ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ;
}
xni = tc_max ( 0.f , tc_mul ( ( tc_add ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dtsave ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
if ( ( xni > 9999.e3f ) ) {
niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_sub ( 9999.e3f , tc_mul ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , odts ) , orho ) ;
}
qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_sub ( tc_sub ( tc_sub ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , orho ) ) ;
nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_sub ( tc_add ( tc_add ( pnr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pnr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pnr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( pnr_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pnr_rcr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pnr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pnr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pnr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pni_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) , orho ) ) ;
xrr = tc_max ( r1 , tc_mul ( ( tc_add ( qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dtsave ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
xnr = tc_max ( r2 , tc_mul ( ( tc_add ( nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dtsave ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
if ( ( xrr > r1 ) ) {
lamr = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_r , crg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , org2 ) , xnr ) , xrr ) ) , obmr ) ;
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , lamr ) ;
if ( ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 2.5e-3f ) ) {
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 2.5e-3f ;
lamr = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
xnr = tc_div ( tc_mul ( tc_mul ( tc_mul ( crg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , org3 ) , xrr ) , tc_pow ( lamr , bm_r ) ) , am_r ) ;
nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_sub ( xnr , tc_mul ( nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , odts ) , orho ) ;
}
else if ( ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < tc_mul ( d0r , 0.75f ) ) ) {
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( d0r , 0.75f ) ;
lamr = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
xnr = tc_div ( tc_mul ( tc_mul ( tc_mul ( crg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , org3 ) , xrr ) , tc_pow ( lamr , bm_r ) ) , am_r ) ;
nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_sub ( xnr , tc_mul ( nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , odts ) , orho ) ;
}
}
else {
qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( - qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ;
nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( - nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odts ) ;
}
qsten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qsten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_sub ( tc_sub ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , orho ) ) ;
qgten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qgten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ( tc_sub ( tc_sub ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( prg_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prg_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_ihm [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , orho ) ) ;
if ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < t_0 ) ) {
tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( ( tc_add ( tc_mul ( tc_mul ( lsub , ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( pri_inu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pri_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_ide [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_iha [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , tc_mul ( tc_mul ( lfus2 , ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( pri_wfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pri_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_rfz [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ) , orho ) , ( tc_sub ( 1 , ifdry ) ) ) ) ;
}
else {
tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( ( tc_add ( tc_mul ( tc_mul ( lfus , ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_sub ( tc_sub ( tc_sub ( ( - prr_sml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rcg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prr_rcs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , tc_mul ( tc_mul ( lsub , ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ) , orho ) , ( tc_sub ( 1 , ifdry ) ) ) ) ;
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( t1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( dt , tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
otemp = tc_div ( 1.f , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
tempc = tc_sub ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) ;
qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 1.e-10f , tc_add ( qv1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( dt , qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( 0.622f , pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_mul ( tc_mul ( r , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 0.622f ) ) ) ) ) ;
rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( tc_div ( rho_not , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
rhof2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = rslf ( pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( tc_div ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , 1.f ) ;
if ( ( tc_abs ( ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) < eps ) ) {
ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
}
diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( 2.11e-5f , tc_pow ( ( tc_div ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) ) , 1.94f ) ) , ( tc_div ( 101325.f , pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
if ( ( tempc >= 0.0f ) ) {
visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_add ( 1.718f , tc_mul ( 0.0049f , tempc ) ) ) , 1.0e-5f ) ;
}
else {
visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_sub ( tc_add ( 1.718f , tc_mul ( 0.0049f , tempc ) ) , tc_mul ( tc_mul ( 1.2e-5f , tempc ) , tempc ) ) ) , 1.0e-5f ) ;
}
vsc2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( tc_div ( rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( lvap0 , tc_mul ( ( tc_sub ( 2106.0f , 4218.0f ) ) , tempc ) ) ;
tcond [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_add ( 5.69f , tc_mul ( 0.0168f , tempc ) ) ) , 1.0e-5f ) , 418.936f ) ;
ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( 1.f , ( tc_mul ( cp , ( tc_add ( 1.f , tc_mul ( 0.887f , qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ) ) ;
lvt2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , orv ) , otemp ) , otemp ) ;
nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 11.1e6f , tc_mul ( ( tc_add ( nwfa1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( nwfaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
if ( ( ( tc_add ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) > r1 ) ) {
rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_add ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 2.f , tc_min ( tc_mul ( ( tc_add ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , nt_c_max ) ) ;
if ( ( ! is_aerosol_aware ) ) {
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = nt_c ;
}
l_qc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
else {
rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 2.f ;
l_qc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( ( tc_add ( qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) > r1 ) ) {
ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_add ( qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r2 , tc_mul ( ( tc_add ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
l_qi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
else {
ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r2 ;
l_qi [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( ( tc_add ( qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) > r1 ) ) {
rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_add ( qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r2 , tc_mul ( ( tc_add ( nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
lamr = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_r , crg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , org2 ) , nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , lamr ) ;
if ( ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 2.5e-3f ) ) {
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 2.5e-3f ;
lamr = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( tc_mul ( tc_mul ( crg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , org3 ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( lamr , bm_r ) ) , am_r ) ;
}
else if ( ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < tc_mul ( d0r , 0.75f ) ) ) {
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( d0r , 0.75f ) ;
lamr = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( tc_mul ( tc_mul ( crg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , org3 ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( lamr , bm_r ) ) , am_r ) ;
}
}
else {
rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r2 ;
l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( ( tc_add ( qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qsten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) > r1 ) ) {
rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_add ( qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qsten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
else {
rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
l_qs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( ( tc_add ( qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qgten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) > r1 ) ) {
rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_add ( qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qgten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = true ;
}
else {
rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = r1 ;
l_qg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
}
if ( ( ! iiwarm ) ) {
for ( k = kts ;
k <= kte ;
k += 1 ) {
smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smob [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smoc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
smod [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
}
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
loga_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sa [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sb [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 14 , ( 1 ) ) * ( 1 ) ] ) ) ;
smod [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( a_ , tc_pow ( smo2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , b_ ) ) ;
}
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
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
lamr = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_r , crg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , org2 ) , nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( 1.f , lamr ) ;
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , lamr ) ;
n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , org2 ) , tc_pow ( lamr , cre [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) ;
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
if ( ( ( ( ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > eps ) ) || ( ( ( ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < ( - eps ) ) && l_qc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) {
clap = tc_div ( ( tc_sub ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , ( tc_add ( 1.f , tc_mul ( lvt2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ;
for ( n = 1 ;
n <= 3 ;
n += 1 ) {
fcd = tc_add ( tc_sub ( tc_mul ( qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_exp ( tc_mul ( lvt2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , clap ) ) ) , qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , clap ) ;
dfcd = tc_add ( tc_mul ( tc_mul ( qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , lvt2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_exp ( tc_mul ( lvt2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , clap ) ) ) , 1.f ) ;
clap = tc_sub ( clap , tc_div ( fcd , dfcd ) ) ;
}
xrc = tc_add ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( clap , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
xnc = 0.f ;
if ( ( xrc > r1 ) ) {
prw_vcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( clap , odt ) ;
if ( ( clap > eps ) ) {
if ( is_aerosol_aware ) {
xnc = tc_max ( 2.f , activ_ncloud ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , w1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
}
else {
xnc = nt_c ;
}
pnc_wcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( 0.5f , ( tc_add ( tc_sub ( xnc , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_abs ( tc_sub ( xnc , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ) , odts ) , orho ) ;
}
else if ( ( ( ( clap < ( - eps ) ) && ( ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < ( - 1.e-6f ) ) ) && is_aerosol_aware ) ) {
tempc = tc_sub ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) ;
otemp = tc_div ( 1.f , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rvs = tc_mul ( rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rvs_p = tc_mul ( tc_mul ( rvs , otemp ) , ( tc_sub ( tc_mul ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , otemp ) , orv ) , 1.f ) ) ) ;
rvs_pp = tc_mul ( rvs , ( tc_add ( tc_add ( tc_mul ( tc_mul ( tc_mul ( otemp , ( tc_sub ( tc_mul ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , otemp ) , orv ) , 1.f ) ) ) , otemp ) , ( tc_sub ( tc_mul ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , otemp ) , orv ) , 1.f ) ) ) , ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( ( - 2.f ) , lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , otemp ) , otemp ) , otemp ) , orv ) ) ) , tc_mul ( otemp , otemp ) ) ) ) ;
gamsc = tc_mul ( tc_div ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tcond [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rvs_p ) ;
alphsc = tc_div ( tc_mul ( tc_div ( tc_mul ( tc_mul ( tc_mul ( 0.5f , ( tc_div ( gamsc , ( tc_add ( 1.f , gamsc ) ) ) ) ) , ( tc_div ( gamsc , ( tc_add ( 1.f , gamsc ) ) ) ) ) , rvs_pp ) , rvs_p ) , rvs ) , rvs_p ) ;
alphsc = tc_max ( 1.e-9f , alphsc ) ;
xsat = ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ;
if ( ( tc_abs ( xsat ) < 1.e-9f ) ) {
xsat = 0.f ;
}
t1_evap = tc_div ( tc_mul ( tc_mul ( 2.f , pi ) , ( tc_sub ( tc_add ( tc_sub ( 1.0f , tc_mul ( alphsc , xsat ) ) , tc_mul ( tc_mul ( tc_mul ( tc_mul ( 2.f , alphsc ) , alphsc ) , xsat ) , xsat ) ) , tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( 5.f , alphsc ) , alphsc ) , alphsc ) , xsat ) , xsat ) , xsat ) ) ) ) , ( tc_add ( 1.f , gamsc ) ) ) ;
dc_star = tc_sqrt ( tc_div ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_div ( tc_mul ( tc_mul ( ( - SD8(0x4000000000000000ul) ) , dt ) , t1_evap ) , ( tc_mul ( 2.f , pi ) ) ) , 4.f ) , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rvs ) , rho_w ) ) ;
idx_d = tc_max ( 1 , tc_min ( ( ( int ) ( tc_mul ( 1.e6f , dc_star ) ) ) , nbc ) ) ;
idx_n = ( ( int ) round ( tc_add ( 1.0f , tc_div ( tc_mul ( F ( nbc ) , tc_log ( tc_div ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , t_nc [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) , nic1 ) ) ) ) ;
idx_n = tc_max ( 1 , tc_min ( idx_n , nbc ) ) ;
if ( ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r_c [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) {
nic = ( ( int ) round ( tc_log10 ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
for ( nn = tc_sub ( nic , 1 ) ;
nn <= tc_add ( nic , 1 ) ;
nn += 1 ) {
n = nn ;
if ( ( ( ( tc_div ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) >= 1.0f ) && ( ( tc_div ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , nn ) ) ) < 10.0f ) ) ) {
break ;
}
}
idx_c = tc_sub ( tc_add ( ( ( int ) ( tc_div ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_pow ( 10.f , n ) ) ) ) , tc_mul ( 10 , ( tc_sub ( n , nic2 ) ) ) ) , ( tc_sub ( n , nic2 ) ) ) ;
idx_c = tc_max ( 1 , tc_min ( idx_c , ntb_c ) ) ;
}
else {
idx_c = 1 ;
}
prw_vcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( tc_mul ( tc_mul ( ( - rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , 0.99f ) , orho ) , odt ) ) ) , prw_vcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
pnc_wcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( ( ( Soft8 ) ( tc_mul ( tc_mul ( tc_mul ( ( - nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , 0.99f ) , orho ) , odt ) ) ) , tc_mul ( tc_mul ( ( - tnc_wev [ 0 + tc_sub ( idx_d , ( 1 ) ) * ( 1 ) + tc_sub ( idx_c , ( 1 ) ) * tc_mul ( 1 , ( ( 100 ) - ( 1 ) + 1 ) ) + tc_sub ( idx_n , ( 1 ) ) * ( 1 * ( ( 100 ) - ( 1 ) + 1 ) * ( ( 37 ) - ( 1 ) + 1 ) ) ] ) , orho ) , odt ) ) ;
}
if ( ( is_aerosol_aware && l_nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) {
l_nwfa [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
}
else {
prw_vcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( - rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , orho ) , odt ) ;
pnc_wcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( - nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , orho ) , odt ) ;
}
qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prw_vcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prw_vcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pnc_wcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nwfaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( nwfaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pnc_wcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prw_vcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_sub ( 1 , ifdry ) ) ) ) ;
rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_mul ( ( tc_add ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( dt , qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
if ( ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] == r1 ) ) {
l_qc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = false ;
}
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 2.f , tc_min ( tc_mul ( ( tc_add ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , nt_c_max ) ) ;
if ( ( ! is_aerosol_aware ) ) {
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = nt_c ;
}
qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 1.e-10f , tc_add ( qv1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( dt , qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( t1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( dt , tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( 0.622f , pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_mul ( tc_mul ( r , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 0.622f ) ) ) ) ) ;
qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = rslf ( pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( tc_div ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , 1.f ) ;
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
if ( ( ( ( ( ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < ( - eps ) ) ) && l_qr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) && ( ( ! ( ( prw_vcd [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 0.f ) ) ) ) ) ) {
tempc = tc_sub ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) ;
otemp = tc_div ( 1.f , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( tc_mul ( rho_not , orho ) ) ;
rhof2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( 2.11e-5f , tc_pow ( ( tc_div ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) ) , 1.94f ) ) , ( tc_div ( 101325.f , pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
if ( ( tempc >= 0.0f ) ) {
visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_add ( 1.718f , tc_mul ( 0.0049f , tempc ) ) ) , 1.0e-5f ) ;
}
else {
visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( tc_sub ( tc_add ( 1.718f , tc_mul ( 0.0049f , tempc ) ) , tc_mul ( tc_mul ( 1.2e-5f , tempc ) , tempc ) ) ) , 1.0e-5f ) ;
}
vsc2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( tc_div ( rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , visco [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( lvap0 , tc_mul ( ( tc_sub ( 2106.0f , 4218.0f ) ) , tempc ) ) ;
tcond [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( ( tc_add ( 5.69f , tc_mul ( 0.0168f , tempc ) ) ) , 1.0e-5f ) , 418.936f ) ;
ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( 1.f , ( tc_mul ( cp , ( tc_add ( 1.f , tc_mul ( 0.887f , qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ) ) ;
rvs = tc_mul ( rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
rvs_p = tc_mul ( tc_mul ( rvs , otemp ) , ( tc_sub ( tc_mul ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , otemp ) , orv ) , 1.f ) ) ) ;
rvs_pp = tc_mul ( rvs , ( tc_add ( tc_add ( tc_mul ( tc_mul ( tc_mul ( otemp , ( tc_sub ( tc_mul ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , otemp ) , orv ) , 1.f ) ) ) , otemp ) , ( tc_sub ( tc_mul ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , otemp ) , orv ) , 1.f ) ) ) , ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( ( - 2.f ) , lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , otemp ) , otemp ) , otemp ) , orv ) ) ) , tc_mul ( otemp , otemp ) ) ) ) ;
gamsc = tc_mul ( tc_div ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tcond [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rvs_p ) ;
alphsc = tc_div ( tc_mul ( tc_div ( tc_mul ( tc_mul ( tc_mul ( 0.5f , ( tc_div ( gamsc , ( tc_add ( 1.f , gamsc ) ) ) ) ) , ( tc_div ( gamsc , ( tc_add ( 1.f , gamsc ) ) ) ) ) , rvs_pp ) , rvs_p ) , rvs ) , rvs_p ) ;
alphsc = tc_max ( 1.e-9f , alphsc ) ;
xsat = tc_min ( ( - 1.e-9f ) , ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
t1_evap = tc_div ( tc_mul ( tc_mul ( 2.f , pi ) , ( tc_sub ( tc_add ( tc_sub ( 1.0f , tc_mul ( alphsc , xsat ) ) , tc_mul ( tc_mul ( tc_mul ( tc_mul ( 2.f , alphsc ) , alphsc ) , xsat ) , xsat ) ) , tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( 5.f , alphsc ) , alphsc ) , alphsc ) , xsat ) , xsat ) , xsat ) ) ) ) , ( tc_add ( 1.f , gamsc ) ) ) ;
lamr = tc_div ( 1.f , ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
if ( ( ( tc_div ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) < 0.95f ) && ( tc_mul ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , orho ) <= 1.e-8f ) ) ) {
prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , orho ) , odts ) ;
}
else {
prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( t1_evap , diffu [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( ( - ssatw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , n0_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rvs ) , ( tc_add ( tc_mul ( t1_qr_ev , tc_pow ( ilamr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , cre [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( t2_qr_ev , vsc2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rhof2 [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_pow ( ( tc_add ( lamr , tc_mul ( 0.5f , fv_r ) ) ) , ( ( - cre [ 0 + tc_sub ( 11 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ) ) ) ;
rate_max = tc_min ( ( tc_mul ( tc_mul ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , orho ) , odts ) ) , tc_mul ( ( tc_sub ( qvs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odts ) ) ;
prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( rate_max ) ) , tc_mul ( prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , orho ) ) ;
if ( ( prr_gml [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 0.0f ) ) {
eva_factor = tc_min ( 1.0f , tc_add ( 0.01f , tc_mul ( ( tc_sub ( 0.99f , 0.01f ) ) , ( tc_div ( tempc , 20.0f ) ) ) ) ) ;
prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , eva_factor ) ;
}
}
pnr_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( ( ( Soft8 ) ( tc_mul ( tc_mul ( tc_mul ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 0.99f ) , orho ) , odts ) ) ) , tc_div ( tc_mul ( prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pnr_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nwfaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( nwfaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , pnr_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_sub ( 1 , ifdry ) ) ) ) ;
rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_mul ( ( tc_add ( qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( dt , qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 1.e-10f , tc_add ( qv1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( dt , qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r2 , tc_mul ( ( tc_add ( nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( dt , nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( t1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( dt , tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( 0.622f , pres [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_mul ( tc_mul ( r , temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( qv [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 0.622f ) ) ) ) ) ;
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
evapprod [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( prv_rev [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( tc_add ( tc_min ( zerod0 , prs_sde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_min ( zerod0 , prg_gde [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ;
rainprod [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( prr_wau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , prr_rcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_scw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_iau [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prg_gcw [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , prs_sci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , pri_rci [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
nstep = 0 ;
for ( int _slice = 1 ;
_slice <= 5 ;
++ _slice ) onstep [ 0 + tc_sub ( _slice , ( 1 ) ) * ( 1 ) ] = 1.0f ;
for ( int _slice = 1 ;
_slice <= 5 ;
++ _slice ) ksed1 [ 0 + tc_sub ( _slice , ( 1 ) ) * ( 1 ) ] = 1 ;
for ( k = tc_add ( kte , 1 ) ;
k >= kts ;
k += ( - 1 ) ) {
vtrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
vtnrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
vtik [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
vtnik [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
vtsk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
vtgk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
vtck [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
vtnck [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.f ;
}
if ( tc_any ( l_qr , kts , kte ) ) {
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
vtr = 0.f ;
rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sqrt ( tc_div ( rho_not , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
if ( ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
lamr = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_r , crg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , org2 ) , nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
vtr = tc_mul ( tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , av_r ) , crg [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] ) , org3 ) , tc_pow ( lamr , cre [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) , ( tc_pow ( ( tc_add ( lamr , fv_r ) ) , ( ( - cre [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
vtrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtr ;
vtr = tc_mul ( tc_mul ( tc_div ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , av_r ) , crg [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] ) , crg [ 0 + tc_sub ( 12 , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( lamr , cre [ 0 + tc_sub ( 12 , ( 1 ) ) * ( 1 ) ] ) ) , ( tc_pow ( ( tc_add ( lamr , fv_r ) ) , ( ( - cre [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] ) ) ) ) ) ;
vtnrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtr ;
}
else {
vtrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtrk [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] ;
vtnrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtnrk [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] ;
}
if ( ( tc_max ( vtrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , vtnrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) > 1.e-3f ) ) {
ksed1 [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] = tc_max ( ksed1 [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , k ) ;
delta_tp = tc_div ( dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( tc_max ( vtrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , vtnrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
nstep = tc_max ( nstep , ( ( int ) ( tc_add ( tc_div ( dt , delta_tp ) , 1.f ) ) ) ) ;
}
}
if ( ( ksed1 [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] == kte ) ) {
ksed1 [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] = tc_sub ( kte , 1 ) ;
}
if ( ( nstep > 0 ) ) {
onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] = tc_div ( 1.f , ( ( F ) ( nstep ) ) ) ;
}
}
if ( tc_any ( l_qc , kts , kte ) ) {
hgt_agl = 0.f ;
for ( k = kts ;
k <= tc_sub ( kte , 1 ) ;
k += 1 ) {
if ( ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r2 ) ) {
ksed1 [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] = k ;
}
hgt_agl = tc_add ( hgt_agl , dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
if ( ( hgt_agl > 500.0f ) ) {
break ;
}
}
for ( k = ksed1 [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] ;
k >= kts ;
k += ( - 1 ) ) {
vtc = 0.f ;
if ( ( ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) && ( w1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < 1.e-1f ) ) ) {
nu_c = tc_min ( 15 , tc_add ( ( ( int ) round ( tc_div ( 1000.e6f , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , 2 ) ) ;
lamc = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , am_r ) , ccg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] ) , ocg1 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
ilamc = tc_div ( 1.f , lamc ) ;
vtc = tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , av_c ) , ccg [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] ) , ocg2 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( ilamc , bv_c ) ) ;
vtck [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtc ;
vtc = tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , av_c ) , ccg [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] ) , ocg1 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( ilamc , bv_c ) ) ;
vtnck [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtc ;
}
}
}
if ( ( ! iiwarm ) ) {
if ( tc_any ( l_qi , kts , kte ) ) {
nstep = 0 ;
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
vti = 0.f ;
if ( ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
lami = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_i , cig [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , oig1 ) , ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmi ) ;
ilami = tc_div ( 1.f , lami ) ;
vti = tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , av_i ) , cig [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , oig2 ) , tc_pow ( ilami , bv_i ) ) ;
vtik [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vti ;
vti = tc_mul ( tc_div ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , av_i ) , cig [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] ) , cig [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( ilami , bv_i ) ) ;
vtnik [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vti ;
}
else {
vtik [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtik [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] ;
vtnik [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtnik [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] ;
}
if ( ( vtik [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 1.e-3f ) ) {
ksed1 [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] = tc_max ( ksed1 [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , k ) ;
delta_tp = tc_div ( dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , vtik [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nstep = tc_max ( nstep , ( ( int ) ( tc_add ( tc_div ( dt , delta_tp ) , 1.f ) ) ) ) ;
}
}
if ( ( ksed1 [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] == kte ) ) {
ksed1 [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] = tc_sub ( kte , 1 ) ;
}
if ( ( nstep > 0 ) ) {
onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] = tc_div ( 1.f , ( ( F ) ( nstep ) ) ) ;
}
}
if ( tc_any ( l_qs , kts , kte ) ) {
nstep = 0 ;
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
vts = 0.f ;
if ( ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
xds = tc_div ( smoc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , smob [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
mrat = tc_div ( 1.f , xds ) ;
ils1 = tc_div ( 1.f , ( tc_add ( tc_mul ( mrat , lam0 ) , fv_s ) ) ) ;
ils2 = tc_div ( 1.f , ( tc_add ( tc_mul ( mrat , lam1 ) , fv_s ) ) ) ;
t1_vts = tc_mul ( tc_mul ( kap0 , csg [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( ils1 , cse [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) ) ;
t2_vts = tc_mul ( tc_mul ( tc_mul ( kap1 , tc_pow ( mrat , mu_s ) ) , csg [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( ils2 , cse [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] ) ) ;
ils1 = tc_div ( 1.f , ( tc_mul ( mrat , lam0 ) ) ) ;
ils2 = tc_div ( 1.f , ( tc_mul ( mrat , lam1 ) ) ) ;
t3_vts = tc_mul ( tc_mul ( kap0 , csg [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( ils1 , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ;
t4_vts = tc_mul ( tc_mul ( tc_mul ( kap1 , tc_pow ( mrat , mu_s ) ) , csg [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] ) , tc_pow ( ils2 , cse [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] ) ) ;
vts = tc_div ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , av_s ) , ( tc_add ( t1_vts , t2_vts ) ) ) , ( tc_add ( t3_vts , t4_vts ) ) ) ;
if ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > ( tc_add ( t_0 , 0.1f ) ) ) ) {
sr = tc_div ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ( tc_add ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ;
vtsk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tc_mul ( vts , sr ) , tc_mul ( ( tc_sub ( 1.f , sr ) ) , vtrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
}
else {
vtsk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( vts , vts_boost [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
}
else {
vtsk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtsk [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] ;
}
if ( ( vtsk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 1.e-3f ) ) {
ksed1 [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] = tc_max ( ksed1 [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , k ) ;
delta_tp = tc_div ( dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , vtsk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nstep = tc_max ( nstep , ( ( int ) ( tc_add ( tc_div ( dt , delta_tp ) , 1.f ) ) ) ) ;
}
}
if ( ( ksed1 [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] == kte ) ) {
ksed1 [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] = tc_sub ( kte , 1 ) ;
}
if ( ( nstep > 0 ) ) {
onstep [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] = tc_div ( 1.f , ( ( F ) ( nstep ) ) ) ;
}
}
if ( tc_any ( l_qg , kts , kte ) ) {
nstep = 0 ;
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
vtg = 0.f ;
if ( ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
vtg = tc_mul ( tc_mul ( tc_mul ( tc_mul ( rhof [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , av_g ) , cgg [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] ) , ogg3 ) , tc_pow ( ilamg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , bv_g ) ) ;
if ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > t_0 ) ) {
vtgk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( vtg , vtrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
else {
vtgk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtg ;
}
}
else {
vtgk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = vtgk [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] ;
}
if ( ( vtgk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 1.e-3f ) ) {
ksed1 [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] = tc_max ( ksed1 [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , k ) ;
delta_tp = tc_div ( dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , vtgk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nstep = tc_max ( nstep , ( ( int ) ( tc_add ( tc_div ( dt , delta_tp ) , 1.f ) ) ) ) ;
}
}
if ( ( ksed1 [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] == kte ) ) {
ksed1 [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] = tc_sub ( kte , 1 ) ;
}
if ( ( nstep > 0 ) ) {
onstep [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] = tc_div ( 1.f , ( ( F ) ( nstep ) ) ) ;
}
}
}
if ( tc_any ( l_qr , kts , kte ) ) {
nstep = ( ( int ) round ( tc_div ( 1.f , onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
for ( n = 1 ;
n <= nstep ;
n += 1 ) {
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
sed_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( vtrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( vtnrk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
k = kte ;
odzq = tc_div ( 1.f , dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_sub ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , dt ) , onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r2 , tc_sub ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , dt ) , onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
for ( k = ksed1 [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ;
k >= kts ;
k += ( - 1 ) ) {
odzq = tc_div ( 1.f , dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_r [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_n [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_add ( rr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_r [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , dt ) , onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r2 , tc_add ( nr [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_n [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , dt ) , onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
}
if ( ( rr [ 0 + tc_sub ( kts , ( kts ) ) * ( 1 ) ] > tc_mul ( r1 , 1000.f ) ) ) {
pptrain = tc_add ( pptrain , tc_mul ( tc_mul ( sed_r [ 0 + tc_sub ( kts , ( kts ) ) * ( 1 ) ] , dt ) , onstep [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ;
}
}
}
if ( tc_any ( l_qc , kts , kte ) ) {
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
sed_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( vtck [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( vtnck [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
for ( k = ksed1 [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] ;
k >= kts ;
k += ( - 1 ) ) {
odzq = tc_div ( 1.f , dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( ( tc_sub ( sed_c [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , orho ) ) ;
ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( ( tc_sub ( sed_n [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , orho ) ) ;
rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_add ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( ( tc_sub ( sed_c [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_c [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , dt ) ) ) ;
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 10.f , tc_add ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( ( tc_sub ( sed_n [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , dt ) ) ) ;
}
}
if ( tc_any ( l_qi , kts , kte ) ) {
nstep = ( ( int ) round ( tc_div ( 1.f , onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
for ( n = 1 ;
n <= nstep ;
n += 1 ) {
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
sed_i [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( vtik [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( vtnik [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
k = kte ;
odzq = tc_div ( 1.f , dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_i [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_sub ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_i [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , dt ) , onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r2 , tc_sub ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , dt ) , onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
for ( k = ksed1 [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ;
k >= kts ;
k += ( - 1 ) ) {
odzq = tc_div ( 1.f , dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_i [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_i [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_n [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_add ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_i [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_i [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , dt ) , onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r2 , tc_add ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_n [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_n [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , dt ) , onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
}
if ( ( ri [ 0 + tc_sub ( kts , ( kts ) ) * ( 1 ) ] > tc_mul ( r1 , 1000.f ) ) ) {
pptice = tc_add ( pptice , tc_mul ( tc_mul ( sed_i [ 0 + tc_sub ( kts , ( kts ) ) * ( 1 ) ] , dt ) , onstep [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) ) ;
}
}
}
if ( tc_any ( l_qs , kts , kte ) ) {
nstep = ( ( int ) round ( tc_div ( 1.f , onstep [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
for ( n = 1 ;
n <= nstep ;
n += 1 ) {
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
sed_s [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( vtsk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
k = kte ;
odzq = tc_div ( 1.f , dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qsten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( qsten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_s [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , onstep [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_sub ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_s [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , dt ) , onstep [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
for ( k = ksed1 [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ;
k >= kts ;
k += ( - 1 ) ) {
odzq = tc_div ( 1.f , dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qsten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qsten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_s [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_s [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , onstep [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_add ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_s [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_s [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , dt ) , onstep [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
}
if ( ( rs [ 0 + tc_sub ( kts , ( kts ) ) * ( 1 ) ] > tc_mul ( r1 , 1000.f ) ) ) {
pptsnow = tc_add ( pptsnow , tc_mul ( tc_mul ( sed_s [ 0 + tc_sub ( kts , ( kts ) ) * ( 1 ) ] , dt ) , onstep [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) ) ;
}
}
}
if ( tc_any ( l_qg , kts , kte ) ) {
nstep = ( ( int ) round ( tc_div ( 1.f , onstep [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
for ( n = 1 ;
n <= nstep ;
n += 1 ) {
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
sed_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( vtgk [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
}
k = kte ;
odzq = tc_div ( 1.f , dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qgten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( qgten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , onstep [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_sub ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( sed_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odzq ) , dt ) , onstep [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
for ( k = ksed1 [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ;
k >= kts ;
k += ( - 1 ) ) {
odzq = tc_div ( 1.f , dzq [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
orho = tc_div ( 1.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
qgten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qgten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_g [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , onstep [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) , orho ) ) ;
rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_add ( rg [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( ( tc_sub ( sed_g [ 0 + tc_sub ( tc_add ( k , 1 ) , ( kts ) ) * ( 1 ) ] , sed_g [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , odzq ) , dt ) , onstep [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) ) ) ;
}
if ( ( rg [ 0 + tc_sub ( kts , ( kts ) ) * ( 1 ) ] > tc_mul ( r1 , 1000.f ) ) ) {
pptgraul = tc_add ( pptgraul , tc_mul ( tc_mul ( sed_g [ 0 + tc_sub ( kts , ( kts ) ) * ( 1 ) ] , dt ) , onstep [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] ) ) ;
}
}
}
if ( ( ! iiwarm ) ) {
for ( k = kts ;
k <= kte ;
k += 1 ) {
xri = tc_max ( 0.0f , tc_add ( qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) ;
if ( ( ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > t_0 ) ) && ( ( xri > 0.0f ) ) ) ) {
qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( xri , odt ) ) ;
ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , odt ) ) ;
qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( xri , odt ) ) ;
niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( ( - ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , odt ) ;
tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( tc_mul ( lfus , ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , xri ) , odt ) , ( tc_sub ( 1 , ifdry ) ) ) ) ;
}
xrc = tc_max ( 0.0f , tc_add ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) ;
if ( ( ( ( temp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < hgfr ) ) && ( ( xrc > 0.0f ) ) ) ) {
lfus2 = tc_sub ( lsub , lvap [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
xnc = tc_add ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ;
qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( xrc , odt ) ) ;
niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( xnc , odt ) ) ;
qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( xrc , odt ) ) ;
ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_sub ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( xnc , odt ) ) ;
tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tc_mul ( tc_mul ( tc_mul ( lfus2 , ocp [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , xrc ) , odt ) , ( tc_sub ( 1 , ifdry ) ) ) ) ;
}
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
t1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( t1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( tten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ;
qv1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 1.e-10f , tc_add ( qv1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qvten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) ;
qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qcten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ;
nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( tc_div ( 2.f , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_min ( tc_add ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( ncten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) , nt_c_max ) ) ;
nwfa1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 11.1e6f , tc_min ( 9999.e6f , ( tc_add ( nwfa1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( nwfaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) ) ) ;
nifa1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( tc_mul ( nain1 , 0.01f ) , tc_min ( 9999.e6f , ( tc_add ( nifa1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( nifaten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) ) ) ;
if ( ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r1 ) ) {
qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
}
else {
nu_c = tc_min ( 15 , tc_add ( ( ( int ) round ( tc_div ( 1000.e6f , ( tc_mul ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) ) ) , 2 ) ) ;
lamc = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_r , ccg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] ) , ocg1 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
xdc = tc_div ( ( tc_add ( tc_add ( bm_r , nu_c ) , 1.f ) ) , lamc ) ;
if ( ( xdc < d0c ) ) {
lamc = tc_div ( cce [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , d0c ) ;
}
else if ( ( xdc > tc_mul ( d0r , 2.f ) ) ) {
lamc = tc_div ( cce [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , ( tc_mul ( d0r , 2.f ) ) ) ;
}
nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( tc_mul ( tc_div ( tc_mul ( tc_mul ( ccg [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) + tc_sub ( nu_c , ( 1 ) ) * tc_mul ( 1 , ( ( 5 ) - ( 1 ) + 1 ) ) ] , ocg2 [ 0 + tc_sub ( nu_c , ( 1 ) ) * ( 1 ) ] ) , qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , am_r ) , tc_pow ( lamc , bm_r ) ) , tc_div ( ( ( Soft8 ) ( nt_c_max ) ) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
}
qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qiten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ;
ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( tc_div ( r2 , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_add ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( niten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) ;
if ( ( qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r1 ) ) {
qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
}
else {
lami = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_i , cig [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , oig1 ) , ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmi ) ;
ilami = tc_div ( 1.f , lami ) ;
xdi = tc_mul ( ( tc_add ( tc_add ( bm_i , mu_i ) , 1.f ) ) , ilami ) ;
if ( ( xdi < 5.e-6f ) ) {
lami = tc_div ( cie [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , 5.e-6f ) ;
}
else if ( ( xdi > 300.e-6f ) ) {
lami = tc_div ( cie [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , 300.e-6f ) ;
}
ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_min ( tc_mul ( tc_div ( tc_mul ( tc_mul ( cig [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , oig2 ) , qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , am_i ) , tc_pow ( lami , bm_i ) ) , tc_div ( SD8(0x4163125300000000ul) , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
}
qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ;
nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( tc_div ( r2 , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_add ( nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( nrten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ) ;
if ( ( qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r1 ) ) {
qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
}
else {
lamr = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_r , crg [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] ) , org2 ) , nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , lamr ) ;
if ( ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 2.5e-3f ) ) {
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 2.5e-3f ;
}
else if ( ( mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < tc_mul ( d0r , 0.75f ) ) ) {
mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_mul ( d0r , 0.75f ) ;
}
lamr = tc_div ( ( tc_add ( tc_add ( 3.0f , mu_r ) , 0.672f ) ) , mvd_r [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ;
nr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( tc_mul ( tc_mul ( crg [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , org3 ) , qr1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , tc_pow ( lamr , bm_r ) ) , am_r ) ;
}
qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qsten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ;
if ( ( qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r1 ) ) {
qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
}
qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_add ( qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , tc_mul ( qgten [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , dt ) ) ;
if ( ( qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r1 ) ) {
qg1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 0.0f ;
}
}
}
void calc_effectrad (TCSPACE F * t1d, TCSPACE F * p1d, TCSPACE F * qv1d, TCSPACE F * qc1d, TCSPACE F * nc1d, TCSPACE F * qi1d, TCSPACE F * ni1d, TCSPACE F * qs1d, TCSPACE F * re_qc1d, TCSPACE F * re_qi1d, TCSPACE F * re_qs1d, int kts, int kte) {
int k ;
TCF ( rho , TCKMAX ) ;
TCF ( rc , TCKMAX ) ;
TCF ( nc , TCKMAX ) ;
TCF ( ri , TCKMAX ) ;
TCF ( ni , TCKMAX ) ;
TCF ( rs , TCKMAX ) ;
F smo2 ;
F smob ;
F smoc ;
F tc0 ;
F loga_ ;
F a_ ;
F b_ ;
Soft8 lamc ;
Soft8 lami ;
bool has_qc ;
bool has_qi ;
bool has_qs ;
int inu_c ;
F g_ratio [ 15 ] = {
24 , 60 , 120 , 210 , 336 , 504 , 720 , 990 , 1320 , 1716 , 2184 , 2730 , 3360 , 4080 , 4896 }
;
has_qc = false ;
has_qi = false ;
has_qs = false ;
for ( k = kts ;
k <= kte ;
k += 1 ) {
rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_div ( tc_mul ( 0.622f , p1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_mul ( tc_mul ( r , t1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ( tc_add ( qv1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 0.622f ) ) ) ) ) ;
rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = ( tc_mul ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) > r1 ) ? tc_mul ( qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) : r1 ;
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 2.f , tc_min ( tc_mul ( nc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , nt_c_max ) ) ;
if ( ( ! is_aerosol_aware ) ) {
nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = nt_c ;
}
if ( ( ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) && ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r2 ) ) ) {
has_qc = true ;
}
ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( r1 , tc_mul ( qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ;
ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = ( tc_mul ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) > r2 ) ? tc_mul ( ni1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) : r2 ;
if ( ( ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) && ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r2 ) ) ) {
has_qi = true ;
}
rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = ( tc_mul ( qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) > r1 ) ? tc_mul ( qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , rho [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) : r1 ;
if ( ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > r1 ) ) {
has_qs = true ;
}
}
if ( has_qc ) {
for ( k = kts ;
k <= kte ;
k += 1 ) {
re_qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 2.49e-6f ;
if ( ( ( rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r1 ) || ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r2 ) ) ) {
continue ;
}
if ( ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] < 100 ) ) {
inu_c = 15 ;
}
else if ( ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] > 1.e10f ) ) {
inu_c = 2 ;
}
else {
inu_c = tc_min ( 15 , tc_add ( ( ( int ) round ( tc_div ( 1000.e6f , nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) ) , 2 ) ) ;
}
lamc = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( nc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , am_r ) , g_ratio [ 0 + tc_sub ( inu_c , ( 1 ) ) * ( 1 ) ] ) , rc [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmr ) ;
re_qc1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 2.51e-6f , tc_min ( tc_sngl ( tc_div ( tc_mul ( SD8(0x3fe0000000000000ul) , ( ( Soft8 ) ( tc_add ( 3.f , inu_c ) ) ) ) , lamc ) ) , 50.e-6f ) ) ;
}
}
if ( has_qi ) {
for ( k = kts ;
k <= kte ;
k += 1 ) {
re_qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 2.49e-6f ;
if ( ( ( ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r1 ) || ( ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r2 ) ) ) {
continue ;
}
lami = tc_pow ( ( tc_div ( tc_mul ( tc_mul ( tc_mul ( am_i , cig [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] ) , oig1 ) , ni [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) , ri [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] ) ) , obmi ) ;
re_qi1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 2.51e-6f , tc_min ( tc_sngl ( tc_div ( tc_mul ( SD8(0x3fe0000000000000ul) , ( ( Soft8 ) ( tc_add ( 3.f , mu_i ) ) ) ) , lami ) ) , 125.e-6f ) ) ;
}
}
if ( has_qs ) {
for ( k = kts ;
k <= kte ;
k += 1 ) {
re_qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = 4.99e-6f ;
if ( ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] <= r1 ) ) {
continue ;
}
tc0 = tc_min ( ( - 0.1f ) , tc_sub ( t1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , 273.15f ) ) ;
smob = tc_mul ( rs [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] , oams ) ;
if ( ( ( bm_s > ( tc_sub ( 2.0f , 1.e-3f ) ) ) && ( bm_s < ( tc_add ( 2.0f , 1.e-3f ) ) ) ) ) {
smo2 = smob ;
}
else {
loga_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sa [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , bm_s ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , bm_s ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , bm_s ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , bm_s ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , bm_s ) , bm_s ) , bm_s ) ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sb [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , bm_s ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , bm_s ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , bm_s ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , bm_s ) , bm_s ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , bm_s ) , bm_s ) , bm_s ) ) ;
smo2 = tc_pow ( ( tc_div ( smob , a_ ) ) , ( tc_div ( 1.f , b_ ) ) ) ;
}
loga_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sa [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sa [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sa [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sa [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ;
a_ = tc_pow ( 10.0f , loga_ ) ;
b_ = tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( tc_add ( sb [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] , tc_mul ( sb [ 0 + tc_sub ( 2 , ( 1 ) ) * ( 1 ) ] , tc0 ) ) , tc_mul ( sb [ 0 + tc_sub ( 3 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 4 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 5 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 6 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 7 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 8 , ( 1 ) ) * ( 1 ) ] , tc0 ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 9 , ( 1 ) ) * ( 1 ) ] , tc0 ) , tc0 ) , tc0 ) ) , tc_mul ( tc_mul ( tc_mul ( sb [ 0 + tc_sub ( 10 , ( 1 ) ) * ( 1 ) ] , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) , cse [ 0 + tc_sub ( 1 , ( 1 ) ) * ( 1 ) ] ) ) ;
smoc = tc_mul ( a_ , tc_pow ( smo2 , b_ ) ) ;
re_qs1d [ 0 + tc_sub ( k , ( kts ) ) * ( 1 ) ] = tc_max ( 5.01e-6f , tc_min ( tc_mul ( 0.5f , ( tc_div ( smoc , smob ) ) ) , 999.e-6f ) ) ;
}
}
}
}
;
#include "hexv842_mpas_thompson_aerosol_column_step_f32_bindings.metalh"
kernel void hexv842_mpas_thompson_aerosol_column_step_f32(
    constant Hexv842MpasThompsonAerosolColumnStepF32Resources &r [[buffer(0)]],
    constant Hexv842MpasThompsonAerosolColumnStepF32Params &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    using namespace ntc;
    device F *qv = (device F *)r.qv;
    device F *qc = (device F *)r.qc;
    device F *qi = (device F *)r.qi;
    device F *qr = (device F *)r.qr;
    device F *qs = (device F *)r.qs;
    device F *qg = (device F *)r.qg;
    device F *ni = (device F *)r.ni;
    device F *nr = (device F *)r.nr;
    device F *th = (device F *)r.th;
    device const F *pii = (device F *)r.pii;
    device const F *p = (device F *)r.p;
    device const F *w = (device F *)r.w;
    device const F *dz = (device F *)r.dz;
    device F *rainprod = (device F *)r.rainprod;
    device F *evapprod = (device F *)r.evapprod;
    device F *re_cloud = (device F *)r.re_cloud;
    device F *re_ice = (device F *)r.re_ice;
    device F *re_snow = (device F *)r.re_snow;
    device F *nc = (device F *)r.aerosol + (0ul);
    device F *nifa = (device F *)r.aerosol + (ulong(params.nz)*ulong(params.columns));
    device F *nwfa = (device F *)r.aerosol + (2ul*ulong(params.nz)*ulong(params.columns));
    device F *rainnc = (device F *)r.rainnc;
    device F *rainncv = (device F *)r.rainncv;
    device F *snownc = (device F *)r.snownc;
    device F *snowncv = (device F *)r.snowncv;
    device F *graupelnc = (device F *)r.graupelnc;
    device F *graupelncv = (device F *)r.graupelncv;
    device F *sr = (device F *)r.sr;
    device const F *ntc = (device F *)r.ntc;
    device const F *muc = (device F *)r.muc;
    device const F *nifa2d = (device F *)r.aerosol + (3ul*ulong(params.nz)*ulong(params.columns));
    device const F *nwfa2d = (device F *)r.aerosol + (3ul*ulong(params.nz)*ulong(params.columns)+ulong(params.columns));
    const int nz = params.nz;
    const int columns = params.columns;
    const F dt = params.dt;
    const int has_reqc = params.has_reqc;
    const int has_reqi = params.has_reqi;
    const int has_reqs = params.has_reqs;
    device const Soft8 *input_tcg_racg = (device const Soft8 *)r.input_tcg_racg;
    device const Soft8 *input_tmr_racg = (device const Soft8 *)r.input_tmr_racg;
    device const Soft8 *input_tcr_gacr = (device const Soft8 *)r.input_tcr_gacr;
    device const Soft8 *input_tmg_gacr = (device const Soft8 *)r.input_tmg_gacr;
    device const Soft8 *input_tnr_racg = (device const Soft8 *)r.input_tnr_racg;
    device const Soft8 *input_tnr_gacr = (device const Soft8 *)r.input_tnr_gacr;
    device const Soft8 *input_tcs_racs1 = (device const Soft8 *)r.input_tcs_racs1;
    device const Soft8 *input_tmr_racs1 = (device const Soft8 *)r.input_tmr_racs1;
    device const Soft8 *input_tcs_racs2 = (device const Soft8 *)r.input_tcs_racs2;
    device const Soft8 *input_tmr_racs2 = (device const Soft8 *)r.input_tmr_racs2;
    device const Soft8 *input_tcr_sacr1 = (device const Soft8 *)r.input_tcr_sacr1;
    device const Soft8 *input_tms_sacr1 = (device const Soft8 *)r.input_tms_sacr1;
    device const Soft8 *input_tcr_sacr2 = (device const Soft8 *)r.input_tcr_sacr2;
    device const Soft8 *input_tms_sacr2 = (device const Soft8 *)r.input_tms_sacr2;
    device const Soft8 *input_tnr_racs1 = (device const Soft8 *)r.input_tnr_racs1;
    device const Soft8 *input_tnr_racs2 = (device const Soft8 *)r.input_tnr_racs2;
    device const Soft8 *input_tnr_sacr1 = (device const Soft8 *)r.input_tnr_sacr1;
    device const Soft8 *input_tnr_sacr2 = (device const Soft8 *)r.input_tnr_sacr2;
    device const Soft8 *input_tpi_qrfz = (device const Soft8 *)r.input_tpi_qrfz;
    device const Soft8 *input_tni_qrfz = (device const Soft8 *)r.input_tni_qrfz;
    device const Soft8 *input_tpg_qrfz = (device const Soft8 *)r.input_tpg_qrfz;
    device const Soft8 *input_tnr_qrfz = (device const Soft8 *)r.input_tnr_qrfz;
    device const Soft8 *input_tpi_qcfz = (device const Soft8 *)r.input_tpi_qcfz;
    device const Soft8 *input_tni_qcfz = (device const Soft8 *)r.input_tni_qcfz;
    device const Soft8 *input_tpi_ide = (device const Soft8 *)r.input_tpi_ide;
    device const Soft8 *input_tps_iaus = (device const Soft8 *)r.input_tps_iaus;
    device const Soft8 *input_tni_iaus = (device const Soft8 *)r.input_tni_iaus;
    device const Soft8 *input_t_efrw = (device const Soft8 *)r.input_t_efrw;
    device const Soft8 *input_t_efsw = (device const Soft8 *)r.input_t_efsw;
    device const Soft8 *input_tpc_wev = (device const Soft8 *)r.input_tpc_wev;
    device const Soft8 *input_tnc_wev = (device const Soft8 *)r.input_tnc_wev;
    device const Soft8 *input_tnr_rev = (device const Soft8 *)r.input_tnr_rev;
    device const F *input_tnccn_act = (device F *)r.input_tnccn_act;
{
int slot = int ( gid ) ; int col = params.first + slot ; if(slot>=params.count)return;
if ( col >= columns ) return ;
ThompsonColumn m ; m.nz=nz; m.tc_wsf=(device F*)r.ws_f+ulong(slot)*93*(nz+1); m.tc_wsd=(device Soft8*)r.ws_d+ulong(slot)*64*(nz+1); m.tc_wsb=(device bool*)r.ws_b+ulong(slot)*7*(nz+1);
m . tcg_racg = input_tcg_racg ;
m . tmr_racg = input_tmr_racg ;
m . tcr_gacr = input_tcr_gacr ;
m . tmg_gacr = input_tmg_gacr ;
m . tnr_racg = input_tnr_racg ;
m . tnr_gacr = input_tnr_gacr ;
m . tcs_racs1 = input_tcs_racs1 ;
m . tmr_racs1 = input_tmr_racs1 ;
m . tcs_racs2 = input_tcs_racs2 ;
m . tmr_racs2 = input_tmr_racs2 ;
m . tcr_sacr1 = input_tcr_sacr1 ;
m . tms_sacr1 = input_tms_sacr1 ;
m . tcr_sacr2 = input_tcr_sacr2 ;
m . tms_sacr2 = input_tms_sacr2 ;
m . tnr_racs1 = input_tnr_racs1 ;
m . tnr_racs2 = input_tnr_racs2 ;
m . tnr_sacr1 = input_tnr_sacr1 ;
m . tnr_sacr2 = input_tnr_sacr2 ;
m . tpi_qrfz = input_tpi_qrfz ;
m . tni_qrfz = input_tni_qrfz ;
m . tpg_qrfz = input_tpg_qrfz ;
m . tnr_qrfz = input_tnr_qrfz ;
m . tpi_qcfz = input_tpi_qcfz ;
m . tni_qcfz = input_tni_qcfz ;
m . tpi_ide = input_tpi_ide ;
m . tps_iaus = input_tps_iaus ;
m . tni_iaus = input_tni_iaus ;
m . t_efrw = input_t_efrw ;
m . t_efsw = input_t_efsw ;
m . tpc_wev = input_tpc_wev ;
m . tnc_wev = input_tnc_wev ;
m . tnr_rev = input_tnr_rev ;
m . tnccn_act = input_tnccn_act ;
m . nt_c = ntc [ col ] ;
m . mu_c = muc [ col ] ;
TCKF ( qv1d , TCKMAX ) ; TCKF ( qc1d , TCKMAX ) ; TCKF ( qi1d , TCKMAX ) ; TCKF ( qr1d , TCKMAX ) ; TCKF ( qs1d , TCKMAX ) ; TCKF ( qg1d , TCKMAX ) ;
TCKF ( ni1d , TCKMAX ) ; TCKF ( nr1d , TCKMAX ) ; TCKF ( nc1d , TCKMAX ) ; TCKF ( nwfa1d , TCKMAX ) ; TCKF ( nifa1d , TCKMAX ) ;
TCKF ( t1d , TCKMAX ) ; TCKF ( p1d , TCKMAX ) ; TCKF ( w1d , TCKMAX ) ; TCKF ( dz1d , TCKMAX ) ;
TCKF ( rainprod1d , TCKMAX ) ; TCKF ( evapprod1d , TCKMAX ) ;
TCKF ( re_qc1d , TCKMAX ) ; TCKF ( re_qi1d , TCKMAX ) ; TCKF ( re_qs1d , TCKMAX ) ;
for ( int k = 0 ;
k < nz ;
k ++ ) {
int idx = k * columns + col ;
qv1d [ k ] = qv [ idx ] ;
qc1d [ k ] = qc [ idx ] ;
qi1d [ k ] = qi [ idx ] ;
qr1d [ k ] = qr [ idx ] ;
qs1d [ k ] = qs [ idx ] ;
qg1d [ k ] = qg [ idx ] ;
ni1d [ k ] = ni [ idx ] ;
nr1d [ k ] = nr [ idx ] ;
nc1d [ k ] = nc [ idx ] ;
nwfa1d [ k ] = nwfa [ idx ] ;
nifa1d [ k ] = nifa [ idx ] ;
t1d [ k ] = th [ idx ] * pii [ idx ] ;
p1d [ k ] = p [ idx ] ;
w1d [ k ] = w [ idx ] ;
dz1d [ k ] = dz [ idx ] ;
rainprod1d [ k ] = 0.0f ;
evapprod1d [ k ] = 0.0f ;
}
F rain = 0.0f , snow = 0.0f , graupel = 0.0f , ice = 0.0f ;
m . mp_thompson ( qv1d , qc1d , qi1d , qr1d , qs1d , qg1d , ni1d , nr1d , nc1d , nwfa1d , nifa1d , t1d , p1d , w1d , dz1d , rain , snow , graupel , ice , rainprod1d , evapprod1d , 1 , nz , dt , 1 , 1 ) ;
rainncv [ col ] = ( ( rain + snow ) + graupel ) + ice ;
rainnc [ col ] = ( ( ( rainnc [ col ] + rain ) + snow ) + graupel ) + ice ;
snowncv [ col ] = snow + ice ;
snownc [ col ] = ( snownc [ col ] + snow ) + ice ;
graupelncv [ col ] = graupel ;
graupelnc [ col ] = graupelnc [ col ] + graupel ;
sr [ col ] = aa_div ( ( ( snow + graupel ) + ice ) , ( rainncv [ col ] + 1.0e-12f ) ) ;
nwfa1d [ 0 ] = nwfa1d [ 0 ] + nwfa2d [ col ] * dt ;
nifa1d [ 0 ] = nifa1d [ 0 ] + nifa2d [ col ] * dt ;
for ( int k = 0 ;
k < nz ;
k ++ ) {
int idx = k * columns + col ;
nc [ idx ] = nc1d [ k ] ;
nwfa [ idx ] = nwfa1d [ k ] ;
nifa [ idx ] = nifa1d [ k ] ;
qv [ idx ] = qv1d [ k ] ;
qc [ idx ] = qc1d [ k ] ;
qi [ idx ] = qi1d [ k ] ;
qr [ idx ] = qr1d [ k ] ;
qs [ idx ] = qs1d [ k ] ;
qg [ idx ] = qg1d [ k ] ;
ni [ idx ] = ni1d [ k ] ;
nr [ idx ] = nr1d [ k ] ;
th [ idx ] = aa_div ( t1d [ k ] , pii [ idx ] ) ;
rainprod [ idx ] = rainprod1d [ k ] ;
evapprod [ idx ] = evapprod1d [ k ] ;
if ( qv1d [ k ] < 0.0f ) {
if ( k + 1 < nz - 2 && k + 1 > 2 ) qv [ idx ] = tc_max ( 1.0e-7f , 0.5f * ( qv [ idx - columns ] + qv [ idx + columns ] ) ) ;
else qv [ idx ] = 1.0e-7f ;
}
}
if ( has_reqc && has_reqi && has_reqs ) {
for ( int k = 0 ;
k < nz ;
k ++ ) {
re_qc1d [ k ] = 2.49e-6f ;
re_qi1d [ k ] = 4.99e-6f ;
re_qs1d [ k ] = 9.99e-6f ;
}
m . calc_effectrad ( t1d , p1d , qv1d , qc1d , nc1d , qi1d , ni1d , qs1d , re_qc1d , re_qi1d , re_qs1d , 1 , nz ) ;
for ( int k = 0 ;
k < nz ;
k ++ ) {
int idx = k * columns + col ;
re_cloud [ idx ] = tc_max ( 2.49e-6f , tc_min ( re_qc1d [ k ] , 50.0e-6f ) ) ;
re_ice [ idx ] = tc_max ( 4.99e-6f , tc_min ( re_qi1d [ k ] , 125.0e-6f ) ) ;
re_snow [ idx ] = tc_max ( 9.99e-6f , tc_min ( re_qs1d [ k ] , 999.0e-6f ) ) ;
}
}
}

}