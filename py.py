## We have two stores,
##    clk_store for clickstream data
##    trx_store for transaction data
## Both are stored at per-minute accuracy


class RetrieverQ(object):
    def __init__(self, store):
        self.store = store

    def retrieve_recent(self, customer_id, curr_time):
        result = []
        for index in range(0, 5):
            val = self.store.retrieve(customer_id, curr_time - 5 + index)
            if val is not None:
                result.append(val)
        return result

        ## Example calling code


## Assume the clk_Store and trx_store are already created

import calendar, time
from retrievers import RetrieverQ

clk_retriever = RetrieverQ(clk_store)
trx_retriever = RetrieverQ(trx_store)

cid = 1

curr_time = calendar.timegm(time.gmtime()) / 60  # minutes since epoch
clk_recent = clk_retriever.retrieve_recent(cid, curr_time)
trx_recent = trx_retriever.retrieve_recent(cid, curr_time)


## Retrieve customers with clickstream events
## In the past we used the code implemented below


def aggregate(recent):
    return len(recent)


def join_data_left(curr_time):
    clk_retriever = RetrieverQ(clk_store)
    trx_retriever = RetrieverQ(trx_store)
    result = []
    for cid in clk_store.cust_ids():
        clk_recent = clk_retriever.retrieve_recent(cid, curr_time)
        trx_recent = trx_retriever.retrieve_recent(cid, curr_time)
        result.append((cid, aggregate(clk_recent), aggregate(trx_recent)))
    return result

    # NOTE:  Minute is sequential number of minutes since the Epoch (1/1/1970)


#
# ----------------------------------    --------------------------------------
# --- Clickstream Data ---              --- Transactions Data ---
# ----------------------------------    --------------------------------------
# CID - MINUTE   - CLICK EVENT            CID - MINUTE   - TRANSACTION
# ----------------------------------    --------------------------------------
# 1   - 25853654 - Confirm                1   - 25853652 - Krogers $243.00
# 1   - 25853653 - Schedule Payment       1   - 25853598 - Starbucks $8.00
# 1   - 25853652 - View recent            1   - 25853590 - Shell Oil $45.00
# 1   - 25853651 - Login                  2   - 25851945 - OSU Bookstore $222
# 1   - 25853650 - Enter password         2   - 25851944 - OSU Bookstore $222
# 1   - 25853649 - Enter user name        3   - 25853651 - Ren's Ice Cream  $5
# 1   - 25853648 - Login                  4   - 25853653 - Geddy's Music  $400
# 2   - 25853650 - Recover Login          4   - 25851002 - Neil's Drums  $1340
# 2   - 25853649 - Login                  5   - 25853400 - Stimpy's Pets  $45


# [25 853 650,  25 853 655]
# id 1 and 2

# (1, 5, 1)
# (2, 1, 0)
