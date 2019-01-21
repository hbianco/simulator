# simulator
Environment to create trading strategies.

## Examples

1. Create and print input rules.
```python
import simulator.simulator as sim
import EcoProject.EcoData as ed

cemsig = sim.Signal(cembi)
cemsig.add_rule({'cover and buy': [
                                       [rsi_vs,'level','above', rsi_s, None],
                                       [rsi_s,'cross','above', rsi_l, 'AND'],
                                       [rsi_vs,'level','above', 30, 'AND'],
                                       [rsi_s,'level','above', 30, 'AND'],
                                       [rsi_l,'level','above', 30, 'AND'],
                                       [cembi,'level','above', ma_vs, 'AND'],
                                  ],
                  'close and sell': [
                                        [rsi_vs,'level','below', rsi_s, None],
                                        [rsi_s, 'cross', 'below', rsi_l,'AND'],
                                        [rsi_vs,'level','below', 70, 'AND'],
                                        [rsi_s,'level','below', 70, 'AND'],
                                        [rsi_l,'level','below', 70, 'AND'],
                                        [cembi,'level','below', ma_vs, 'AND'],
                                        [rsi_vs,'level','below', 30, 'OR'],
                                        [rsi_s,'level','below', 30, 'AND'],
                                        [rsi_l,'cross','below', 30, 'AND'],
                                    ],
                   'buy': [
                                   [cembi,'level','below', ma_vs, None],
                                   [ma_s, 'level', 'above', ma_l,'AND']
                           ],
                   'exit short': [
                                        [cembi,'cross','above', ma_vs, None],
                                        [ma_s, 'level', 'above', ma_vs,'AND'],
                                        [ma_l, 'level', 'above', ma_s,'AND'],
                                        [rsi_vs, 'level', 'below', 40,'AND'],
                                        [rsi_s, 'level', 'below', 40,'AND'],
                                        [rsi_l, 'level', 'below', 40,'AND']
                                 ]
                })

cemsig.eval_strat()
print(cemsig)
```
![alt text](/static/strategy_display.png)


2. Graph strategy with signals and PnL.
```python
import EcoData as ed

ed.graph(cembi, cemsig.pnl, multiple_series=True,
        trading_signal=cemsig.trigger,
         chart_size=(15,10))
```
![alt text](/static/strategy_with_pnl.png)


3. Graph strategy with signals and RSI.
![alt text](/static/strategy_signals.png)
