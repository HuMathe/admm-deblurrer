import sys
import time
import termcolor
import terminaltables

class Timing:
    timing_record = {}
    context = {}
    def trigger(item, func, *args, **kwargs):
        if item not in Timing.timing_record:
            Timing.timing_record[item] = {'evoke':0, 'time': 0}
        local_time = time.time()
        result = func(*args, **kwargs)
        local_time = time.time() - local_time
        Timing.timing_record[item]['evoke'] += 1
        Timing.timing_record[item]['time'] += local_time
        return result
    
    def set_ckpt(name, is_entry=False):
        if is_entry:
            name = f'-> {name}'
        def deco(func):
            def inner(*args, **kwargs):
                if is_entry:
                    Timing.timing_record = {}
                    Timing.context = {}
                result = Timing.trigger(name, func, *args, **kwargs)
                if is_entry:
                    Timing.print_result()
                return result
            return inner
        return deco
    
    def print_result():
        print(terminaltables.AsciiTable([
            ['name', 'evoke_num', 'cum_time'],
            *[[termcolor.colored(name, 'light_blue'), v['evoke'], f'{v["time"]:.3f}'] 
              for name, v in sorted(Timing.timing_record.items(), key=(lambda kv: kv[1]['time']), reverse=True)],
            *[[termcolor.colored(name, 'light_yellow'), v['evoke'], f'{v["time"]:.3f}'] 
              for name, v in sorted(Timing.context.items(), key=(lambda kv: kv[1]['time']), reverse=True)],
        ]).table, file=sys.stderr)

    def recording(rec_name):
        class TimingDoll:
            def __exit__(self, *args):
                local_time = time.time() - self.start_time
                rec = Timing.context.get(rec_name, {'evoke':0, 'time':0})
                rec['evoke'] += 1
                rec['time'] += local_time
                Timing.context[rec_name] = rec
            def __enter__(self, *args):
                self.start_time = time.time()
        return TimingDoll()
