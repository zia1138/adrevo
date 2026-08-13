import random
import collections
import json

from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


# ---------- Workload simulator (from txn_simulator.py) ----------

class Workload:
    """
    Constructor for taking in transactions and representing them as
    (read/write, key, position, txn_len)
    """
    def __init__(self, workload_json, debug=False, verify=False):
        self.workload = list(json.loads(workload_json).values())
        self.num_txns = len(self.workload)
        self.debug = debug
        self.verify = verify
        self.txns = []
        self.only_hot_keys = False
        self.hot_keys_thres = 100
        self.hot_keys = set()
        self.hot_keys_map = {}
        self.sorted_len = None
        self.median_len = 0
        self.conflict_blocks = []
        self.conflict_blocks_map = {}
        self.m = 0

        self.get_txns()

    def get_txns(self):
        key_freqs = {}
        len_map = {}
        lens = []
        for txn in self.workload:
            txn_ops = []
            ops = txn.split(" ")
            txn_len = len(ops)
            count = 0
            tmp1 = None
            tmp2 = None
            tmp3 = None
            for i in range(len(ops)):
                op = ops[i]
                if op != "*":
                    vals = op.split("-")
                    if len(vals) != 2:
                        print(op, vals)
                    assert len(vals) == 2
                    if self.only_hot_keys and int(vals[1]) > self.hot_keys_thres:
                        tmp1 = vals[0]
                        tmp2 = vals[1]
                        tmp3 = i + 1
                        continue
                    else:
                        count += 1
                    txn_ops.append((vals[0], vals[1], i + 1, len(ops)))
                    if vals[1] not in key_freqs:
                        key_freqs[vals[1]] = 1
                    else:
                        key_freqs[vals[1]] += 1
                    if len(ops) not in len_map:
                        len_map[len(ops)] = 1
                    else:
                        len_map[len(ops)] += 1
                    lens.append(len(ops))
            if count == 0 and self.only_hot_keys:
                txn_ops.append((tmp1, tmp2, tmp3, len(ops)))
            self.txns.append(txn_ops)
        if self.debug:
            print(self.txns)
        self.num_txns = len(self.txns)

    def insert_key_map(self, key, key_map, op_type, key_start, key_end, txn_id):
        index = len(key_map[key]) - 1
        for op in key_map[key]:
            (_, s, e, _) = key_map[key][index]
            if e <= key_end:
                if s <= key_start:
                    index += 1
                    break
            elif s <= key_start:
                index += 1
                break
            index -= 1
        if index == -1:
            (_, s, e, _) = key_map[key][0]
            if key_end < e and key_start < s:
                key_map[key].insert(0, (op_type, key_start, key_end, txn_id))
            else:
                key_map[key].append((op_type, key_start, key_end, txn_id))
        else:
            key_map[key].insert(index, (op_type, key_start, key_end, txn_id))
        if self.debug:
            print("insert: ", index, key, key_start, key_end, key_map[key])

    def find_earliest_read(self, key, key_map, txn_id):
        if key_map[key][-1][3] == txn_id:
            print("TXN_ID")
            return key_map[key][-1][1]
        else:
            if self.debug:
                print(key, key_map[key], txn_id)
            index = len(key_map[key]) - 1
            while key_map[key][index][0] == "r":
                if index == -1:
                    break
                index -= 1
        if self.debug:
            print("index: ", index)
        if index == -1:
            index = 0
        else:
            index = key_map[key][index][2] + 1
        return index

    def get_opt_seq_cost(self, txn_seq):
        """
        Gets the makespan of a given sequence of transactions.

        Returns:
            Value representing the makespan (time to execute given schedule)
        """
        if self.debug:
            print("seq: ", txn_seq)
        key_map = {}
        prev_txn = txn_seq[0]
        total_cost = 0
        txn_id = 0
        cost_map = {}
        for i in range(len(txn_seq)):
            time = i
            txn = self.txns[txn_seq[i]]
            txn_start = 1
            txn_total_len = 0
            max_release = 0
            cost = 0
            for j in range(len(txn)):
                (op_type, key, pos, txn_len) = txn[j]
                if key in key_map:
                    key_start = 0
                    if key_map[key][-1][0] == "w" or op_type == "w":
                        key_start = key_map[key][-1][2] + 1
                    else:
                        key_start = self.find_earliest_read(key, key_map, txn_id)
                    txn_start = max(txn_start, key_start - pos + 1)
                    if self.debug:
                        print(key, key_start, pos, txn_start)
                    max_release = max(max_release, key_start - 1)
                txn_total_len = txn_len
            txn_end = txn_start + txn_total_len - 1
            cost = txn_end - total_cost
            if txn_end <= total_cost:
                cost = 0
            if cost in cost_map:
                cost_map[cost] += 1
            else:
                cost_map[cost] = 1
            total_cost += cost
            if self.debug:
                print(txn, txn_start, txn_end, max_release, cost, total_cost)

            curr_txn = txn_seq[i]
            prev_txn = curr_txn
            if self.debug:
                print(txn_start, txn_end, max_release, cost)

            for j in range(len(txn)):
                (op_type, key, pos, txn_len) = txn[j]
                key_start = txn_start + pos - 1
                if key in key_map:
                    if key_map[key][-1][0] == "w" or op_type == "w":
                        self.insert_key_map(key, key_map, op_type, key_start, key_start, txn_id)
                    else:
                        self.insert_key_map(key, key_map, op_type, key_start, key_start, txn_id)
                else:
                    key_map[key] = [(op_type, key_start, key_start, txn_id)]
            if self.debug:
                print(key_map)
            txn_id += 1
        if self.debug:
            print(total_cost)

        od = collections.OrderedDict(sorted(cost_map.items()))
        return total_cost


# ---------- Evolvable scheduling code ----------

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using greedy cost sampling strategy.

    Returns:
        A permutation of transaction indices.
    """
    def get_greedy_cost_sampled(num_samples, sample_rate):
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(0, workload.num_txns)]
        remaining_txns.remove(start_txn)
        running_cost = workload.txns[start_txn][0][3]

        for i in range(0, workload.num_txns - 1):
            min_cost = 100000
            min_relative_cost = 10
            min_txn = -1
            holdout_txns = []
            done = False

            sample = random.random()
            if sample > sample_rate:
                idx = random.randint(0, len(remaining_txns) - 1)
                t = remaining_txns[idx]
                txn_seq.append(t)
                remaining_txns.pop(idx)
                continue

            for j in range(0, num_samples):
                idx = 0
                if len(remaining_txns) > 1:
                    idx = random.randint(0, len(remaining_txns) - 1)
                else:
                    done = True
                t = remaining_txns[idx]
                holdout_txns.append(remaining_txns.pop(idx))
                if workload.debug:
                    print(remaining_txns, holdout_txns)
                txn_len = workload.txns[t][0][3]
                test_seq = txn_seq.copy()
                test_seq.append(t)
                cost = workload.get_opt_seq_cost(test_seq)
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
                if done:
                    break
            assert(min_txn != -1)
            running_cost = min_cost
            txn_seq.append(min_txn)
            holdout_txns.remove(min_txn)
            remaining_txns.extend(holdout_txns)

            if workload.debug:
                print("min: ", min_txn, remaining_txns, holdout_txns, txn_seq)
        if workload.debug:
            print(txn_seq)
            print(len(set(txn_seq)))
        assert len(set(txn_seq)) == workload.num_txns

        overall_cost = workload.get_opt_seq_cost(txn_seq)

        return overall_cost, txn_seq

    _, schedule = get_greedy_cost_sampled(10, 1.0)
    return schedule


def get_schedules():
    """Return one transaction schedule for each built-in workload."""
    workloads = [
        Workload(WORKLOAD_1),
        Workload(WORKLOAD_2),
        Workload(WORKLOAD_3),
    ]
    return [get_best_schedule(workload, 10) for workload in workloads]


if __name__ == "__main__":
    schedules = get_schedules()
    print(f"Schedules: {schedules}")
