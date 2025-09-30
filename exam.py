# a = 100
# b = 10
# print(b or a)
#
# c = [1, 2, 3]
# print(c[:-1])
#
# proportions = [0] * 3
# print(proportions[1])
#
# def info(args):
#     for name, value in args.items():
#         print(f"{name} = {value}")
#
#
# dict1 = {
#     "acc": 0.700,
#     "pcs": 0.699,
#     "recall": 0.599,
#     "f1": 0.799
# }
#
# def dictcopy(args: dict):
#     dict_clone = {}
#     for key, value in args.items():
#         dict_clone[key] = value
#     return dict_clone
#
# print(dictcopy(dict1))
# print(dict1.values())

dict2 = dict(
    classifier='mobilenet_v1',
    guide='resnet50film',
    crcnet='crcnet'
)

dict2.update(model_config=dict(
    hello=1,
    world=2
))

print(dict2)

