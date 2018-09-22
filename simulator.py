# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:42:01 2018

@author: herve.biancotto
"""

import pandas as pd
import numpy as np

MAP_ACTION={
            'buy':'b',
            'sell':'s',
            'cover and buy':'cb',
            'close and sell':'cs',
            'exit long':'el',
            'exit short':'es',
            }

REV_MAP={
            'b':'buy',
            's':'sell',
            'cb':'cover and buy',
            'cs':'close and sell',
            'el':'exit long',
            'es':'exit short',
            }

def cross_signal(s1, s2, direction):
    
    try:
        s1, s2 = test_input(s1, s2)
    except (TypeError, IndexError) as e:
        raise e
    
    cross_sig = s1.copy()
    
    # lag series
    s1_lag = s1.shift(1).fillna(method='pad')
    s1_lag.iloc[0] = s1.iloc[0] # initial value
    s2_lag = s2.shift(1).fillna(method='pad')
    s2_lag.iloc[0] = s2.iloc[0] # inital value
    sig_d = s1.copy()
    sig_y = s1_lag.copy()
    
    # level signal
    sig_d[s1.values >= s2.values] = 1.0
    sig_d[s1.values < s2.values] = -1.0
    sig_y[s1_lag.values >= s2_lag.values] = 1.0
    sig_y[s1_lag.values < s2_lag.values] = -1.0
    
    # Create crossing signal
    if direction=='above':
        cross_sig[(sig_d.values==1.0) & (sig_y==-1.0)] = 1.0 
        cross_sig[(sig_d.values==-1.0) | (sig_y==1.0)] = 0.0 
    elif direction=='below':
        cross_sig[(sig_d.values==-1.0) & (sig_y==1.0)] = 1.0
        cross_sig[(sig_d.values==1.0) | (sig_y==-1.0)] = 0.0
    
    return cross_sig
    

def level_signal(s1, s2, direction):
    
    try:
        s1, s2 = test_input(s1, s2)
    except (TypeError, IndexError) as e:
        raise e
    
    level_sig = s1.copy()
    
    # Create level signal
    if direction=='above':
        level_sig[s1.values>s2.values] = 1.0 
        level_sig[(s1.values<s2.values)] = 0.0 
    elif direction=='below':
        level_sig[(s1.values<s2.values)] = 1.0
        level_sig[(s1.values>s2.values)] = 0.0
    
    return level_sig


def test_input(s1, s2):
    # Check whether second series is a single value or a dataframe
    if  isinstance(s2, pd.DataFrame):
        s2 = pd.Series(s2.iloc[:,0], index=s1.index)
    elif  isinstance(s2, (int, float)):
        s2 = pd.Series(s2, index=s1.index)
    elif  isinstance(s2, (list, np.array)):
        if len(s2)!=len(s1):
            raise IndexError('the two series must be of equal length. Got {0:d} and {1:d}'.format(len(s1), len(s2)))
        s2 = pd.Series(s2, index=s1.index)
    elif not isinstance(s2, (pd.DataFrame, pd.Series)):
        raise TypeError('Data must be a pandas object. Got {0}'.format(type(s2)))
    
    # Check data are dataseries
    if  isinstance(s1, pd.DataFrame):
        s1 = pd.Series(s1.iloc[:,0], index=s1.index)
    if not isinstance(s1, (pd.DataFrame, pd.Series)):
        raise TypeError('Data must be a pandas object. Got {0}'.format(type(s1)))
   
    # Check signal is same length as data
    if s1.shape[0]!=s2.shape[0]:
        raise IndexError('Data and signal should have same row dimension. Got {0} and {1}'
                         .format(s1.shape[0], s2.shape[0]))
    
    return s1, s2


def strategy_signal(*signal):
    """
    Take a list of individual signals as an input and package them to create a single signal for the strategy as output.
    """
    # mapping for signal creation
    func = {'cross': cross_signal,
            'level': level_signal}
    
    # series for signal
    signal_s = pd.DataFrame(np.zeros(len(signal[0][0])), index=signal[0][0].index, columns=['s0'])
    final_signal = signal_s.copy()
    
    logical = [] # list of 'AND' boolean logics between signals.
    and_indices = []
    
    #Create signal series, organise by logic
    for i, s in enumerate(signal):
        s1, trig, cond, s2, log = s
        s = func[trig](s1, s2, cond)
        signal_s['s' + str(i)] = s
        if log==None or log=='AND':
            and_indices.append(i)
        elif log=='OR':
            logical.append(and_indices)
            and_indices = [i]
    logical.append(and_indices)
    
    # Group according to logic
    for j, and_block in enumerate(logical):
        if len(and_block)>1:
            final_signal['s' + str(j)] =  signal_s.iloc[:,and_block].all(axis=1)
        else:
            final_signal['s' + str(j)] =  signal_s.iloc[:,and_block]
    
    # Create final single signal
    final_signal = final_signal.any(axis=1)
    
    return final_signal
    

class Signal:
    def __init__(self,  series=None, rule=None):
        self.rules = {v:[] for v in MAP_ACTION.values()}
        if rule is not None:
            self.add_rule(rule)
        self.series = series
        if self.series is None:
            self.index = None
        else:
            self.index = series.index # analysis time frame
        self.strategy = None # series of long and shorts
        self.trigger = None # series of trigger points
        self.pnl = None # pnl
        self._transaction_cost = False
        
    def add_rule(self, rule):
        try:
            if not isinstance(rule, dict): 
                rule = dict(rule)
        except TypeError:
            raise TypeError(f'Cannot convert {type(rule)} to dictionary')
        
        for k,v in rule.items():
            if isinstance(v[0], list):
                self.rules[MAP_ACTION[k]]+=v
                if self.index is None:
                    self.index = v[0][0].index
            else:
                self.rules[MAP_ACTION[k]].append(v)
                if self.index is None:
                    self.index = v[0].index
    
    def delete_rule(self, rule_to_del):
        # :param rule <dict>: key = rule name, value = sub-rule index
        try:
            if not isinstance(rule_to_del, dict): rule_to_del = dict(rule_to_del)
        except TypeError:
            raise TypeError(f'Cannot convert {type(rule_to_del)} to dictionary')
        
        for k,idx in rule_to_del.items():
            if isinstance(idx,int):
                del self.rules[MAP_ACTION[k]][idx]
            elif isinstance(idx,list):
                for i in sorted(idx)[::-1]:
                    del self.rules[MAP_ACTION[k]][i]
            elif isinstance(idx,str) and idx=='all':
                self.rules[MAP_ACTION[k]] = []
    
    def alter_rule(self, rule):
        # :param rule <dict>: key = rule name, value = [sub_rule index, new value]
        try:
            if not isinstance(rule, dict): rule = dict(rule)
        except TypeError:
            raise TypeError(f'Cannot convert {type(rule)} to dictionary')
        
        for k, v in rule.items():
            self.rules[MAP_ACTION[k]][v[0]][3] = v[1]
    
    def __repr__(self):
        for k,v in self.rules.items():
            if v!=[]: 
                print(f'* {REV_MAP[k]} : \n')
                for i, sub_r in enumerate(v):
                    if isinstance(sub_r[3], pd.DataFrame): label = sub_r[3].columns[0]
                    elif isinstance(sub_r[3], pd.Series): label = sub_r[3].name
                    else: label = str(sub_r[3])
                    if i==0:
                        print(f'    {i} - {sub_r[0].columns[0]} {sub_r[1]} {sub_r[2]} {label}\n')
                    else:
                        print(f'    {i} - {sub_r[4]} {sub_r[0].columns[0]} {sub_r[1]} {sub_r[2]} {label}\n')
        return ''
    
    @classmethod
    def show_actions(cls):
        for k in MAP_ACTION.keys():
            print(f'* {k}\n')

    def create_strategy_(self, **kwargs):
        """
        Creates the strategy series, i.e. a series of -1, 0, 1 to indicate buy/sell/flat.
        It does not compute the P&L of the strategy.
        """
        # Input:
        #=======
        # signal: <list> of [action, [series1, trigger {level or cross}, condition, series2, link to previous signal {and, or}]]
        # start_bias : <str> in the list {'off', 'long', 'short'}. Inital trade.
        # on_day_trading: <bool> start the trade on the day the signal is triggered. By default, False.
        # ret_trigger: <bool> return the trigger series as well as the strategy series.
        
        # Start with trade on or off
        trade = 'off'
        if "start_bias"  in kwargs:
            trade = kwargs['start_bias']
            
        # On day trading
        on_day_trading = False
        if "on_day_trading" in kwargs:
            on_day_trading = kwargs['on_day_trading']
        
        # Return the signal trigger series
        inplace = True
        if "inplace" in kwargs:
            inplace = kwargs['inplace']
        
        strategy = pd.Series(0, index=self.index).to_frame()
        strategy.columns=['strategy']
            
        for k, s in self.rules.items():
            if s!=[]:
                sig = strategy_signal(*s)
                strategy[k] = sig
        
        # Build the strategy signal from individual signals
        signal = []
        trigger = []
        t = 0
        stop = 0
        df_iter = strategy.iterrows() # Efficient iteration with generator
        for row in df_iter:
            if not on_day_trading: # Trading happens the following day a signal is triggered. REFACTORING NEEDED
                # Register trading signal
                s, trig, t  = self.append_trading_signal(trade, t, stop)
                signal.append(s)
                trigger.append(trig)
                # Generate trading signal
                trade, t, stop = self.generate_trading_signal(row[1], trade, t)
                    
            else: # Case trade should be implemented on the day of signal trigger.
                # Generate signal
                trade, t, stop = self.generate_trading_signal(row[1], trade, t)
                # Trading happens on the day
                s, trig, t  = self.append_trading_signal(trade, t, stop)
                signal.append(s)
                trigger.append(trig)
        
        strategy['strategy'] = signal
        strategy['trigger'] = trigger
        
        if inplace:
            self.strategy = strategy['strategy']
            self.trigger = strategy['trigger']
        else:
            return strategy['strategy'], strategy['trigger']
    
    
    def append_trading_signal(self, trade, trigger_sig, stop):
        if trade=='long':
            signal = 1
            if trigger_sig:
                trigger = 1
                trigger_sig = False
            else:
                trigger = 0 
        elif trade=='short':
            signal = -1
            if trigger_sig:
                trigger = -1
                trigger_sig = False
            else:
                trigger = 0
        elif trade=='off':
            signal = 0
            if trigger_sig:
                trigger = stop
                trigger_sig = False
            else:
                trigger = 0
        return signal, trigger, trigger_sig
    
    
    def generate_trading_signal(self, signal_list, trade, t):
        stop = 0
        if signal_list.get('b', False) and trade=='off':
            trade='long'
            t = True
        elif signal_list.get('s', False) and trade=='off':
            trade='short'
            t = True
        elif signal_list.get('cb', False) and (trade=='short' or trade=='off'):
            trade='long'
            t = True
        elif signal_list.get('cs', False) and (trade=='long' or trade=='off'):
            trade='short'
            t = True
        elif signal_list.get('el', False) and trade=='long':
            trade='off'
            t = True
            stop = -1
        elif signal_list.get('es', False) and trade=='short':
            trade='off'
            t = True
            stop = 1
        return trade, t, stop
    
        
    def compute_pnl_(self, size=1.0, currency=1, cumulative=True, **kwargs):
        """
        Computes the PnL of a trading strategy given an asset to trade on.
        """
        series = self.series
        strategy = self.strategy
        
        if isinstance(series, pd.DataFrame):
            series = pd.Series(series.iloc[:,0])
        if isinstance(strategy, pd.DataFrame):
            strategy = pd.Series(strategy.iloc[:,0])
        
        # Deal with keyword arguments
        transaction_cost = False
        if 'transaction_cost' in kwargs:
            transaction_cost = kwargs['transaction_cost']
        
        inplace = True
        if 'inplace' in kwargs:
            inplace = kwargs['inplace']
            
        if 'currency' in kwargs:    
            if len(kwargs['currency'])==3:
                ccy = 'USD' + kwargs['currency']
            elif len(kwargs['currency'])==6: 
                ccy = kwargs['currency']
            currency = bp.getHistoricalData(ccy, 'px last', self.index[0], self.index[-1]).values
        
        trigger = self.trigger
        
        if transaction_cost:
            series = series - trigger * transaction_cost * series
        
        if not isinstance(currency, int):
            series = currency * series
        
        pnl = series.copy()
        
        ret = series / series.shift(1) - 1
        pnl = ret * strategy + 1 
        pnl[0] = size
        pnl.name = 'PnL'
        
        cumpnl = pnl.cumprod()
        cumpnl.name = 'Cum_PnL'
        
        if cumulative:
            if inplace:
                self.pnl = cumpnl
            else:
                return cumpnl
        else:
            if inplace:
                self.pnl = pnl
            else:
                return pnl
            
    def eval_strat(self, **args):
        if self.strategy is not None:
            self.strategy = None
            self.trigger = None
            self.pnl = None
        
        self.create_strategy_()
        self.compute_pnl_()
    
    def change_series(self, new_series):
        if self.series is None:
            return
        if self.series.shape[0]!= new_series.shape[0]:
            raise IndexError('The new series has {new_series.shape[0]} observations. Expecting {self.series.shape[0]}')
            
        self.series = new_series
        
    @property
    def transaction_cost(self):
        return self._transaction_cost
    @transaction_cost.setter
    def transaction_cost(self, new_value):
        self._transaction_cost = new_value
    
    
if __name__=='__main__':
    import bloompy as bp
    sp = bp.getHistoricalData('SPX Index','px last','20171231')
    import sys
    PATH = r'P:\PythonScripts'
    if PATH not in sys.path:
        sys.path.append(r'P:\PythonScripts')
    import technical.technical as ta
    sp_rsi = ta.rsi(sp, 14)
    # Create signal object
    s = Signal(sp)
    
    s.add_rule({'close and sell': [[sp,'cross','below', 2850, None], \
                                   [sp_rsi, 'level', 'below', 50,'OR']]})
    s.add_rule({'cover and buy': [[sp,'level','above', 2800, None], \
                                  [sp_rsi,'level','above', 70, 'AND'],\
                                  [sp_rsi,'cross','above', 35, 'OR']]})
    s.add_rule({'exit long': [sp,'cross','above', 2830, None]})
    
    print(s)
    
    s.create_strategy_()
    s.compute_pnl_()
    print(s.pnl[-1]-1)
    
    s.alter_rule({'exit long':[0,2855]})
    print(s)
    
    s.eval_strat()
    print(s.pnl[-1]-1)