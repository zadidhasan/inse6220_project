from render_human_pose import render_pose

# pose = [
#     [0, 0, 0],
#     [-0.1285, 0.0105, -0.0507],
#     [0.0277, 0.251, -0.4071],
#     [0.0115, 0.6402, -0.1885],
#     [0.1285, -0.0105, 0.0507],
#     [0.258, 0.221, -0.3221],
#     [0.183, 0.6034, -0.1038],
#     [-0.0153, -0.2269, -0.0387],
#     [0.0001, -0.4691, -0.1334],
#     [0.0145, -0.5108, -0.2333],
#     [0.0029, -0.6233, -0.2024],
#     [0.1219, -0.4233, -0.0753],
#     [0.2896, -0.1993, -0.0496],
#     [0.235, -0.1718, -0.2943],
#     [-0.1273, -0.4067, -0.147],
#     [-0.2696, -0.1651, -0.1659],
#     [-0.1328, -0.0807, -0.3603]
# ]

pose = [[-0.040488436818122864, -0.4019530117511749, 0.11888356506824493],
[0.05595859885215759, -0.840811550617218, 0.2042658030986786],
[0.047947946935892105, -0.8617779016494751, 0.3658313751220703],
[0.020511265844106674, -0.8330249190330505, 0.42568734288215637], 
[0.17020410299301147, -0.4321160614490509, -0.003467700444161892],
[0.19123110175132751, -0.845253586769104, -0.21254795789718628],
[0.2019595354795456, -0.906083345413208, -0.05873492360115051],
[0.2166096717119217, -0.9028857350349426, 0.011462968774139881],
[0.033093083649873734, 0.23899085819721222, -0.03825313597917557],
[-0.040470801293849945, 0.46192678809165955, -0.12728160619735718],
[-0.10826341062784195, 0.5416032671928406, -0.08233863115310669],
[-0.0762677863240242, 0.6273683309555054, -0.16065005958080292],
[0.09386990964412689, 0.4082161784172058, -0.08460012078285217],
[0.21518678963184357, 0.16523174941539764, -0.01620427705347538],
[0.2725660502910614, -0.05368172749876976, 0.08853397518396378],
[0.20920288562774658, -0.03900166600942612, 0.16716104745864868],
[0.31384336948394775, -0.1365792453289032, 0.13228723406791687],
[-0.16608838737010956, 0.3757179379463196, -0.1279878169298172],
[-0.3585974872112274, 0.16993039846420288, -0.127348393201828],
[-0.31079602241516113, 0.17371869087219238, 0.1293732076883316],
[-0.3208203911781311, 0.27441704273223877, 0.11692759394645691],
[-0.23620839416980743, 0.2088869959115982, 0.23823972046375275]]


joint_links = [
    [0, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [8, 11],
    [11, 12],
    [12, 13],
    [8, 14],
    [14, 15],
    [15, 16],
    [0, 1],
    [1, 2],
    [2, 3],
    [0, 4],
    [4, 5],
    [5, 6],
]

import numpy as np

render_pose(pose=np.array(pose), joint_links=joint_links)