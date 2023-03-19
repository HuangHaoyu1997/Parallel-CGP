import numpy as np


def decision_func(props_value):
    t2t_price = props_value[0]
    open_price = props_value[1]
    ask_info = np.array(props_value[2])
    bid_info = np.array(props_value[3])
    quote_price = props_value[4]
    quote_quantity = props_value[5]
    quote_side = props_value[6]
    quote_type = props_value[7]
    hold_list = np.array(props_value[8])
    
    bound = 0.05
    dp = 0.04 # 0.08
    quote_list = []
    if ask_info.shape[0] == 0 and bid_info.shape[0] == 0:
        return quote_list
    else:
        # print(quote_type, quote_side, quote_price, quote_quantity, (1-bound)*open_price, (1+bound)*open_price)
        if quote_type == 0: # 限价单
            if quote_side == 1 and bid_info.shape[0] != 0: # 当前单是卖单
                if quote_price >= int((1 + bound) * open_price): # 涨停
                    if quote_price <= bid_info[-1][0]: # 小于买单里。， 的最高价，有可能成交
                        index = bid_info[:,0]>=quote_price # ,1.1*open_price
                        quantity = bid_info[index,1].sum()
                        x = ow.s_g.props[ow.s_g.props_name['quote']].set_value([0, 1, np.abs(quantity), int((1+dp)*open_price), 0, -1]).quote
                        quote_list.append(x)
                        # quote_list.append([0, 1, 1.02*open_price, quantity, 0,-1])
                if quote_price < int((1 - bound) * open_price):  # 涨停
                    
                    if quote_price <= bid_info[-1][0]:  # 低于买单里的最高价，有可能成交
                        # 这种情况，为未来留下处理空间，所以我们这里要挂买单
                        q1 = 0  # 超过边界的卖单
                        q2 = 0  # 没超过边界的买单
                        for ai in ask_info:
                            if ai[0] <= int((1 - bound) * open_price):  # ai[0] > int((1 + bound) * open_price) and
                                q1 += ai[1]
                        for bi in bid_info:
                            if bi[0] >= int((1 - bound) * open_price):  # ai[0] > int((1 + bound) * open_price) and
                                q2 += bi[1]
                        quantity = -(q1 + q2) + quote_quantity  # q1为负，q2为正
                        x = ow.s_g.props[ow.s_g.props_name['quote']].set_value(
                            [0, 0, np.abs(quantity), int((1 - dp) * open_price), 0, -1]).quote
                        if quantity != 0:
                            quote_list.append(x)

            if quote_side == 0 and ask_info.shape[0] != 0: # 0 bid,想要买入的价格 1 ask,想要卖出的价格
                if quote_price < int((1 - bound) * open_price):
                    if quote_price >= ask_info[-1][0]: # 高于卖单里的最低价，有可能成交
                        index = ask_info[:,0] <= quote_price #, 0.9*open_price
                        quantity = ask_info[index,1].sum()
                        x = ow.s_g.props[ow.s_g.props_name['quote']].set_value(
                            [0, 0, np.abs(quantity), int((1-dp)*open_price), 0, -1]).quote
                        quote_list.append(x)
                        # quote_list.append([0, 0, 0.98*open_price, quantity, 0, -1])
                if quote_price > int((1 + bound) * open_price):
                    # 这种情况，为未来留下处理空间，所以我们这里要挂卖单
                    
                    if quote_price >= 1.0 * ask_info[-1][0]:  # 高于卖单里的最低价，有可能成交
                        q1 = 0  # 没超过边界的卖单
                        q2 = 0  # 超过边界的买单
                        for ai in ask_info:
                            if ai[0] <= int((1 + bound) * open_price):  # ai[0] > int((1 + bound) * open_price) and
                                q1 += ai[1]
                        for bi in bid_info:
                            if bi[0] >= int((1 + bound) * open_price):  # ai[0] > int((1 + bound) * open_price) and
                                q2 += bi[1]
                        quantity = q1 + q2 + quote_quantity  # q1为负，q2为正
                        x = ow.s_g.props[ow.s_g.props_name['quote']].set_value(
                            [0, 1, np.abs(quantity), int((1 + dp) * open_price), 0, -1]).quote
                        if quantity != 0:
                            quote_list.append(x)
        elif quote_type == 1:  # 市价单,以实时价格买入或卖出
            if quote_side == 1 and bid_info.shape[0] != 0:  # 卖单,找买单里的最高价去成交
                q1 = 0
                q2 = 0
                q3 = 0
                for bi in bid_info:
                    if bi[0] > int((1 + bound) * open_price):
                        q1 += bi[1]
                    if bi[0] >= int((1 - bound) * open_price) and bi[0] <= int((1 + bound) * open_price):
                        q2 += bi[1]
                    else:
                        q3 += bi[1]
                if abs(q1) > 0:
                    x = ow.s_g.props[ow.s_g.props_name['quote']].set_value(
                        [0, 1, np.abs(q1), int((1 + dp) * open_price), 0, -1]).quote
                    quote_list.append(x)
                if abs(q3) > 0 and quote_quantity - abs(q2) > 0:
                    # 发买单，对冲之后了能会超的卖单，但要注意已有卖单量
                    q4 = 0
                    for ai in ask_info:
                        if ai[0] <= int((1 - dp) * open_price):
                            q4 += ai[1]
                    x = ow.s_g.props[ow.s_g.props_name['quote']].set_value(
                        [0, 0, np.abs(quote_quantity - abs(q2) + abs(q4)), int((1 - dp) * open_price), 0, -1]).quote
                        
        # elif quote_type == 1: # 市价单,以实时价格买入或卖出
        #     if quote_side == 1: # 卖单,找买单里的最高价去成交
        #         max_bid_price = bid_info[:,0].max()
        #         if max_bid_price >= (1+bound)*open_price:
        #             # print('here 3')
        #             index = bid_info[:,0] >= (1+bound)*open_price
        #             quantity = bid_info[index,1].sum()
        #             x = ow.s_g.props[ow.s_g.props_name['quote']].set_value([0, 1, np.abs(quantity), (1+dp)*open_price, 0, -1]).quote
                    quote_list.append(x)

            elif 1==0 and quote_side == 0 and ask_info.shape[0] != 0: # 买单,找卖单里的最低价去成交
                q1 = 0
                q2 = 0
                q3 = 0
                for ai in ask_info:
                    if ai[0] < int((1 - bound) * open_price):
                        q1 += ai[1]
                    if ai[0] >= int((1 - bound) * open_price) and ai[0] <= int((1 + bound) * open_price):
                        q2 += ai[1]
                    else:
                        q3 += ai[1]
                if abs(q1) > 0:
                    x = ow.s_g.props[ow.s_g.props_name['quote']].set_value(
                        [0, 0, np.abs(q1), int((1 - dp) * open_price), 0, -1]).quote
                    quote_list.append(x)
                if abs(q3) > 0 and quote_quantity - abs(q2) > 0:
                    # 发卖单，对冲之后了能会超的买单，但要注意已有买单量
                    q4 = 0
                    for bi in bid_info:
                        if bi[0] >= int((1 + dp) * open_price):
                            q4 += bi[1]
                    x = ow.s_g.props[ow.s_g.props_name['quote']].set_value(
                        [0, 1, np.abs(quote_quantity - abs(q2) + abs(q4)), int((1 + dp) * open_price), 0, -1]).quote
                    quote_list.append(x)
                # min_ask_price = ask_info[:,0].min()
                # if min_ask_price <= (1-bound)*open_price:
                #     # print('here 4')
                #     index = ask_info[:,0]<=(1-bound)*open_price
                #     quantity = ask_info[index,1].sum()
                #     x = ow.s_g.props[ow.s_g.props_name['quote']].set_value(
                #         [0, 0, np.abs(quantity), (1-dp)*open_price, 0, -1]).quote
                #     quote_list.append(x)
        
        if len(hold_list) > 0: # 持有一些单子
            for p, q in hold_list:
                if q < 0: # ask卖单
                    if p <= int((1 + bound) * open_price) and p >= int((1 - bound) * open_price): # 卖单价低于实时价，应该挂出
                        x = ow.s_g.props[ow.s_g.props_name['quote']].set_value([0, 1, np.abs(q), p, 0, -1]).quote
                        quote_list.append(x)
                        # quote_list.append([0, 1, p, np.abs(q), 0, -1])
                if q > 0: # 这是一个买单bid
                    if p >= int((1 - bound) * open_price) and p <= int((1 + bound) * open_price): # 买单价高于实时市场价，应该挂出
                        x = ow.s_g.props[ow.s_g.props_name['quote']].set_value([0, 0, np.abs(q), p, 0, -1]).quote
                        quote_list.append(x)
                        # quote_list.append([0, 0, p, np.abs(q), 0, -1])
    return quote_list

