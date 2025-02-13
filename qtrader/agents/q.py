import numpy as np
from datetime import datetime
from qtrader.agents.base import BaseAgent
from qtrader.rlflow.persistence import BasePersistenceProvider
from typing import Dict, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


class QAgent(BaseAgent):
    ACTION_DO_NOTHING = "DO_NOTHING"
    ACTION_BUY_1X = "BUY_1X"
    ACTION_BUY_2X = "BUY_2X"
    ACTION_BUY_3X = "BUY_3X"
    ACTION_SELL_1X = "SELL_1X"
    ACTION_SELL_2X = "SELL_2X"
    ACTION_SELL_3X = "SELL_3X"
    ACTION_CLOSE_POSITION = "CLOSE_POSITION"

    ACTIONS = [
        ACTION_DO_NOTHING,
        ACTION_BUY_1X, ACTION_BUY_2X, ACTION_BUY_3X,
        ACTION_SELL_1X, ACTION_SELL_2X, ACTION_SELL_3X,
        ACTION_CLOSE_POSITION
    ]

    def __init__(self, name: str, pprovider: BasePersistenceProvider,
                 expl_max=1, expl_min=0.01, expl_decay=0.9,  # exploration
                 invest_pct=0.05, invest_multiplier=(1, 1.5, 2),  # how much to invest in an order
                 retrain_days=7, exp_memory_days=120,  # retraining and exp replay
                 model_lr=0.0001, model_l2_reg=0.001, model_dropout=0.5,  # model related
                 rl_gamma=0.9,  # rl related,

                 no_learn=False
                 ):
        super(QAgent, self).__init__()

        self.name = name

        self.pprovider = pprovider
        assert isinstance(self.pprovider, BasePersistenceProvider)

        self.expl_min = expl_min
        self.expl_decay = expl_decay
        self.expl_rate = expl_max  # needs to be loaded and persisted

        self.invest_pct = invest_pct
        self.invest_multiplier = invest_multiplier
        self.reduce_multiplier = (1 / 4, 2 / 4, 3 / 4)

        self.retrain_days = retrain_days
        self.exp_memory_days = exp_memory_days
        self.retrain_last_dt = None  # needs to be loaded and persisted
        self.model_target_update_last_dt = None  # needs to be loaded and persisted

        self.model_lr = model_lr
        self.model_l2_reg = model_l2_reg
        self.model_dropout = model_dropout

        self.rl_gamma = rl_gamma

        self.model_name = "QAgent-Model.h5"
        self.model = None
        self.model_target = None
        self._load_model()
        self._load_model_target()
        self.update_model_target = False
        self.num_learns = 0  # need to be loaded and persisted

        self.state_list = "QAgent-State"
        self.state_prefix = f"{self.state_list}"

        self.no_learn = no_learn
        self._load_config()

    def _load_config(self):
        try:
            c = self.pprovider.load_dict("QAgent-Params")
            self.expl_rate = c.get('expl_rate', self.expl_rate)
            self.retrain_last_dt = datetime.fromisoformat(c.get('retrain_last_dt', datetime(1000, 1, 1).isoformat()))
            self.model_target_update_last_dt = datetime.fromisoformat(c.get('model_target_update_last_dt', datetime(1000, 1, 1).isoformat()))
            self.num_learns = c.get('num_learns', 0)

        except:
            pass

    def _save_config(self):
        self.pprovider.persist_dict(
            name='QAgent-Params', obj={
                'retrain_last_dt': self.retrain_last_dt.isoformat(),
                'model_target_update_last_dt': self.model_target_update_last_dt.isoformat(),
                'expl_rate': self.expl_rate,
                'num_learns': self.num_learns
            }
        )

    def _possible_actions(self, symbol: str, state: dict):
        account_value = state['state_global']['account']['value']
        pos = state['state_symbol'][symbol]['position']
        price = state['state_symbol'][symbol]['ohlcv']['close'][-1]
        invest_amounts = [account_value * self.invest_pct * im for im in self.invest_multiplier]
        stock_amounts = [np.floor(ia / price) for ia in invest_amounts]

        pactions = np.ones(len(self.ACTIONS))
        # is close position possible
        pactions[self.ACTIONS.index(self.ACTION_CLOSE_POSITION)] = 1 if pos else 0
        # are BUYs/SELLS possible
        if not pos:
            pactions[self.ACTIONS.index(self.ACTION_BUY_1X)] = 1 if stock_amounts[0] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_BUY_2X)] = 1 if stock_amounts[1] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_BUY_3X)] = 1 if stock_amounts[2] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_SELL_1X)] = 1 if stock_amounts[0] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_SELL_2X)] = 1 if stock_amounts[1] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_SELL_3X)] = 1 if stock_amounts[2] > 0 else 0

        elif pos and pos['size'] > 0:  # BUY position
            pactions[self.ACTIONS.index(self.ACTION_BUY_1X)] = 1 if stock_amounts[0] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_BUY_2X)] = 1 if stock_amounts[1] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_BUY_3X)] = 1 if stock_amounts[2] > 0 else 0

            size = abs(pos['size'])
            reduce_position_amounts = [np.floor(size * rm) for rm in self.reduce_multiplier]
            pactions[self.ACTIONS.index(self.ACTION_SELL_1X)] = 1 if reduce_position_amounts[0] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_SELL_2X)] = 1 if reduce_position_amounts[1] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_SELL_3X)] = 1 if reduce_position_amounts[2] > 0 else 0

        elif pos and pos['size'] < 0:  # SELL position
            pactions[self.ACTIONS.index(self.ACTION_SELL_1X)] = 1 if stock_amounts[0] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_SELL_2X)] = 1 if stock_amounts[1] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_SELL_3X)] = 1 if stock_amounts[2] > 0 else 0

            size = abs(pos['size'])
            reduce_position_amounts = [np.floor(size * rm) for rm in self.reduce_multiplier]
            pactions[self.ACTIONS.index(self.ACTION_BUY_1X)] = 1 if reduce_position_amounts[0] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_BUY_2X)] = 1 if reduce_position_amounts[1] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_BUY_3X)] = 1 if reduce_position_amounts[2] > 0 else 0

        return pactions

    def _shape_action(self, a: dict, symbol: str, state: dict):
        account_value = state['state_global']['account']['value']
        pos = state['state_symbol'][symbol]['position']
        price = state['state_symbol'][symbol]['ohlcv']['close'][-1]
        invest_amounts = [account_value * self.invest_pct * im for im in self.invest_multiplier]
        stock_amounts = [np.floor(ia / price) for ia in invest_amounts]
        size = abs(pos['size']) if pos else 0
        reduce_position_amounts = [np.floor(size * rm) for rm in self.reduce_multiplier]

        if a['action_private'].startswith("BUY"):
            if not pos or pos['size'] > 0:  # open BUY position # add to BUY position
                a['action'] = "BUY"
                a['type'] = 'limit'
                a['price'] = price
                if a['action_private'] == self.ACTION_BUY_1X:
                    a['size'] = stock_amounts[0]
                if a['action_private'] == self.ACTION_BUY_2X:
                    a['size'] = stock_amounts[1]
                if a['action_private'] == self.ACTION_BUY_3X:
                    a['size'] = stock_amounts[2]

            elif pos and pos['size'] < 0:  # reduce SELL position
                a['action'] = "BUY"
                a['type'] = 'market'
                if a['action_private'] == self.ACTION_BUY_1X:
                    a['size'] = reduce_position_amounts[0]
                if a['action_private'] == self.ACTION_BUY_2X:
                    a['size'] = reduce_position_amounts[1]
                if a['action_private'] == self.ACTION_BUY_3X:
                    a['size'] = reduce_position_amounts[2]

        elif a['action_private'].startswith("SELL"):
            if not pos or pos['size'] < 0:  # open SELL position # add to SELL position
                a['action'] = "SELL"
                a['type'] = 'limit'
                a['price'] = price
                if a['action_private'] == self.ACTION_SELL_1X:
                    a['size'] = stock_amounts[0]
                if a['action_private'] == self.ACTION_SELL_2X:
                    a['size'] = stock_amounts[1]
                if a['action_private'] == self.ACTION_SELL_3X:
                    a['size'] = stock_amounts[2]

            elif pos and pos['size'] > 0:  # reduce BUY position
                a['action'] = "SELL"
                a['type'] = 'market'
                if a['action_private'] == self.ACTION_SELL_1X:
                    a['size'] = reduce_position_amounts[0]
                if a['action_private'] == self.ACTION_SELL_2X:
                    a['size'] = reduce_position_amounts[1]
                if a['action_private'] == self.ACTION_SELL_3X:
                    a['size'] = reduce_position_amounts[2]

        elif a['action_private'] == self.ACTION_CLOSE_POSITION:
            a['action'] = self.ACTION_CLOSE_POSITION

        else:
            a['action'] = a['action_private']

        return a

    def act(self, state: dict) -> Dict:
        symbols = state['state_global']['symbols']

        actions = {}
        for sy in symbols:
            pa = self._possible_actions(sy, state)

            # exploration or model agent prediction
            if self.model is None \
                or np.random.rand() < self.expl_rate:  # or (pa.sum() == 1 and pa[0] == 1) \
                # create random action
                p = np.random.rand(len(self.ACTIONS)) * pa
                a = self.ACTIONS[np.argmax(p)]
                actions[sy] = {'action_private': a, 'method': 'random', 'actions_possible': pa}

            else:
                # use model
                columns, _, _, ex = self._generate_example(sy, state)
                ex = pd.DataFrame([ex], columns=columns)

                ct = self.pprovider.load_obj(name='QAgent-Scaler')
                ex = ct.transform(ex)
                pa[pa == 0] = -np.inf
                p = self.model.predict(ex)[0] * pa
                a = self.ACTIONS[np.argmax(p)]
                actions[sy] = {'action_private': a, 'method': 'model', 'actions_possible': pa, 'predictions': p}

        # shape actions
        for sy in symbols:
            actions[sy] = self._shape_action(actions[sy], sy, state)

        return actions

    def feedback(self, state: dict, action: dict, reward: dict, state_future: dict) -> None:
        # find valid state for symbols
        symbols_valid = [
            sy for sy in state['state_global']['symbols']
            if sy in state_future['state_global']['symbols']
        ]

        if len(symbols_valid) == 0:
            return

        state['state_global']['symbols'] = symbols_valid

        # adjust reward
        account_value = state['state_global']['account']['value']
        for sy in state['state_global']['symbols']:
            reward[sy]['r'] = 100 * (reward[sy]['v_curr'] - reward[sy]['v_prev']) / (self.invest_pct * account_value)

        # update state
        state['reward'] = reward
        state['state_future'] = state_future

        # persist updated state
        dt = datetime.fromisoformat(state['state_global']['account']['current_datetime']).strftime("%Y%m%d%H")
        state_name = f"{self.state_prefix}-{self.name}-{dt}"
        self.pprovider.persist_dict(name=state_name, obj=state)

    def ready_to_learn(self, state: dict) -> bool:
        if self.no_learn: return False

        run_learn = False
        self.update_model_target = False

        states = sorted(self.pprovider.list(prefix=self.state_list), reverse=True)
        dt = datetime.fromisoformat(state['state_global']['account']['current_datetime'])
        if len(states) >= self.exp_memory_days \
               and (not self.retrain_last_dt
                    or (dt - self.retrain_last_dt).days > self.retrain_days
                    or (dt - self.retrain_last_dt).days < 0):
            self.expl_rate = max(self.expl_min, self.expl_rate*self.expl_decay)
            self.retrain_last_dt = dt
            self.num_learns += 1
            run_learn = True

        if run_learn \
                and (not self.model_target_update_last_dt
                     or (dt - self.model_target_update_last_dt).days > self.retrain_days*5
                     or (dt - self.model_target_update_last_dt).days < 0):
            self.model_target_update_last_dt = dt
            self.update_model_target = True

        if run_learn:
            self._save_config()
            return True

        return False

    def _generate_example(self, symbol: str,  state: dict) -> Optional[tuple]:
        columns = []
        columns_norm = []
        columns_std = []
        ex = []

        # POSITION
        cdt = datetime.fromisoformat(state['state_global']['account']['current_datetime'])
        account_value = state['state_global']['account']['value']
        pos = state['state_symbol'][symbol]['position']
        trade = state['state_symbol'][symbol]['trade']
        price = state['state_symbol'][symbol]['ohlcv']['close'][-1]

        # TIME
        columns.append('WeekdayX'); ex.append(np.cos(np.pi * cdt.weekday() / 3));
        columns.append('WeekdayY'); ex.append(np.sin(np.pi * cdt.weekday() / 3));

        # POSITION
        columns.append('Symbol_Value'); ex.append(np.log((self.invest_pct*account_value)/price))  #; columns_norm.append('Symbol_Value')
        columns.append('Position_Open'); ex.append(1 if pos else 0)
        columns.append("Position_Direction"); ex.append(0 if not pos else pos['size']/abs(pos['size']))
        columns.append('Position_Profit'); ex.append(0 if not pos else 100*pos['profit'] / account_value); columns_std.append('Position_Profit')
        columns.append('Position_Size'); ex.append(0 if not pos else (abs(pos['size']) * price) / account_value); # columns_norm.append('Position_Size')
        columns.append('Trade_Num_Orders'); ex.append(0 if not pos else len(trade)); columns_norm.append('Trade_Num_Orders')
        columns.append('Trade_In_Time'); ex.append(
            (cdt - datetime.fromisoformat(trade[0]['datetime'])).days
            if pos else
            -1 * (cdt - datetime.fromisoformat(trade[-1]['datetime'])).days
            if trade else 0
        ); columns_std.append('Trade_In_Time')

        # BRIDGE BANDS
        bb = state['state_symbol'][symbol]['bridge_bnds']
        for i in range(1, 11):
            columns.append(f'BB_W_{i - 1}'); ex.append(bb['bridge_bands_width'][-i])
            columns.append(f'BB_Pos_{i - 1}'); ex.append(bb['bridge_bands_pos'][-i])
            # columns.append(f'BB_HExp_{i - 1}'); ex.append(bb['hurst_exp'][-i]); columns_norm.append(f'BB_HExp_{i - 1}')

        # MODEL
        m = state['state_symbol'][symbol]['model_ind']
        columns.append(f'Model_ACC'); ex.append(m['params']['eval']['acc']); columns_norm.append(f'Model_ACC')
        columns.append(f'Model_AUC'); ex.append(m['params']['eval']['auc']); columns_norm.append(f'Model_AUC')
        for i in range(len(m['preds']['p'])):
            if not np.isnan(m['preds']['target'][i]):  # hist prediction
                columns.append(f"Model_P_{i}"); ex.append(
                    -1 * np.log2(1e-6 + abs(m['preds']['target'][i] - m['preds']['p'][i])) - 1
                )

            else:
                columns.append(f"Model_P_{i}"); ex.append(2 * (m['preds']['p'][i] - 0.5))

        # TRENDLINES
        trendlines = state['state_symbol'][symbol]['trendlines']
        pp = trendlines['pivot_points']
        lines = trendlines['lines']
        data = trendlines['data']

        for i in range(5):
            if i < len(pp['close']):
                columns.append(f'PP_{i}'); ex.append(1 - pp['close'][i] / price); columns_std.append(f'PP_{i}');
                columns.append(f'PP_Age_{i}'); ex.append(pp['ind'][i]); columns_norm.append(f'PP_Age_{i}');

            else:
                columns.append(f'PP_{i}'); ex.append(0); columns_std.append(f'PP_{i}');
                columns.append(f'PP_Age_{i}'); ex.append(len(data['close'])); columns_norm.append(f'PP_Age_{i}');

        n_last = 10
        for i in range(4):
            if i < len(lines):
                c_ln = 1 - np.array(data[f'line_{i}'][-n_last:]) / np.array(data['close'][-n_last:])
                columns.append(f"TL_L{i}_Age"); ex.append(lines[i]['pp'][-1]); columns_norm.append(f"TL_L{i}_Age");
                for k, tl_diff in enumerate(c_ln):
                    columns.append(f"TL_L{i}_Lag{k}"); ex.append(tl_diff); columns_std.append(f"TL_L{i}_Lag{k}");

            else:
                columns.append(f"TL_L{i}_Age"); ex.append(len(data['close'])); columns_norm.append(f"TL_L{i}_Age");
                for k in range(n_last):
                    columns.append(f"TL_L{i}_Lag{k}"); ex.append(0); columns_std.append(f"TL_L{i}_Lag{k}");

        return columns, columns_norm, columns_std, ex

    def learn(self) -> None:
        def reward_clipping(r):
            return 2 * ((1 / (1 + np.exp(-r / 1))) - 0.5)

        states = sorted(self.pprovider.list(prefix=self.state_list), reverse=True)

        columns = None
        columns_norm = None
        columns_std = None
        examples_state = []
        examples_state_future = []
        reward = []
        action = []
        action_future = []

        p = np.exp(-np.linspace(0, len(states)-1, len(states))/self.exp_memory_days*2)
        states_sel = np.random.choice(states, size=self.exp_memory_days, p=p / p.sum(), replace=False)
        for sf in states_sel:
            state = self.pprovider.load_dict(name=sf)
            state_future = state['state_future']
            for sy in state['state_global']['symbols']:
                columns, columns_norm, columns_std, ex = self._generate_example(sy, state)
                examples_state.append(ex)
                columns, columns_norm, columns_std, ex = self._generate_example(sy, state_future)
                examples_state_future.append(ex)
                reward.append(state['reward'][sy]['r'])
                action.append(state['action'][sy])
                action_future.append(state_future['action'][sy])

        examples_state = pd.DataFrame(examples_state, columns=columns)
        examples_state_future = pd.DataFrame(examples_state_future, columns=columns)
        reward = reward_clipping(np.array(reward))

        null_check = examples_state.isnull().any(axis=1)
        if null_check.sum() > 0:
            print('LEARN: Null values in example_state.')
            print(f'LEARN: {states_sel[null_check]}')

        null_check = examples_state_future.isnull().any(axis=1)
        if null_check.sum() > 0:
            print('LEARN: Null values in examples_state_future.')
            print(f'LEARN: {states_sel[null_check]}')

        ct = ColumnTransformer([
            ('stdscaler', StandardScaler(), columns_norm),
            ('stdscaler_std', StandardScaler(with_mean=False), columns_std)
        ], remainder='passthrough')
        ct.fit(examples_state)
        examples_state = ct.transform(examples_state)
        examples_state_future = ct.transform(examples_state_future)
        self.pprovider.persist_obj(name="QAgent-Scaler", obj=ct)

        if self.model is None:
            self._create_model(input_size=examples_state.shape[1], output_size=len(self.ACTIONS),
                               lr=self.model_lr, l2_reg=self.model_l2_reg, dropout=self.model_dropout)
            self.model_target = self.model

        q_values = self.model.predict(examples_state)
        q_values_future = self.model.predict(examples_state_future)
        q_values_future_t = self.model_target.predict(examples_state_future)

        X = []
        y = []
        cls_w = np.ones(len(self.ACTIONS)) + 1e-8
        sample_w = []
        for i in range(examples_state.shape[0]):
            X.append(examples_state[i])
            pa = np.array(action_future[i]['actions_possible'], dtype=np.float)
            pa[pa == 0] = -np.inf

            a = action[i]['action_private']
            a_index = self.ACTIONS.index(a)
            r = reward[i]

            q = q_values[i]
            q_a = q[a_index]
            q_f_action_index = np.argmax(q_values_future[i] * pa)
            q_t_t = q_values_future_t[i][q_f_action_index]
            td = abs(r + self.rl_gamma * q_t_t - q_a)

            q[a_index] = r + self.rl_gamma * q_t_t
            y.append(q)
            cls_w[a_index] += 1
            sample_w.append(td + 1e-4)

        # class weight calc
        # cls_w = (cls_w.sum() / (cls_w.size * cls_w))
        # cls_w = {i: cls_w[i] for i in range(cls_w.size)}

        # sample weight calc
        sample_w = np.array(sample_w)
        sample_w = sample_w / sample_w.sum()  # * sample_w.size
        subsample = np.random.choice(sample_w.size, size=self.exp_memory_days, p=sample_w)

        self.model.fit(x=np.array(X)[subsample],
                       y=np.array(y)[subsample],
                       batch_size=64,
                       epochs=100,
                       validation_split=0.8,
                       # class_weight=cls_w,
                       sample_weight=sample_w[subsample] / sample_w[subsample].sum() * subsample.size,
                       shuffle=True,
                       verbose=2,
                       callbacks=self._model_callbacks()
        )
        self._load_model()
        print(f"EXPL. RATE: {self.expl_rate}")

        if self.update_model_target:
            self._load_model_target()
            print(f"MODEL TARGET UPDATED!")

        self._write_avg_reward(states[:self.retrain_days], epoch=self.num_learns)

    def _create_model(self, input_size: int, output_size: int,
                      lr: float = 0.001, l2_reg: float = 0.001, dropout: float = 0.5):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, input_shape=(input_size,),
                                  activation='sigmoid',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(256,
                                  activation='sigmoid',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(128,
                                  activation='sigmoid',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(output_size, activation='linear')
        ])
        self.model.compile(
            loss="mse",  # tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        )

    def _load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.pprovider.root_join(self.model_name))

        except Exception as e:
            self.model = None
            print(f"Model load error: {e}")

    def _load_model_target(self):
        try:
            self.model_target = tf.keras.models.load_model(self.pprovider.root_join(self.model_name))

        except Exception as e:
            self.model_target = None
            print(f"Model Target load error: {e}")

    def _model_callbacks(self):
        cb_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.pprovider.root_join(self.model_name),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        )

        cb_early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=1,
            mode="min"
        )

        return [
            cb_model_checkpoint,
            # cb_early_stopping,
            tf.keras.callbacks.TensorBoard(self.pprovider.root_join('QAgent-Tensorboard'))
        ]

    def _write_avg_reward(self, state, epoch):
        avg_reward_writer = tf.summary.create_file_writer(self.pprovider.root_join('QAgent-Tensorboard/reward'))
        rwd = []
        for sf in state:
            s = self.pprovider.load_dict(name=sf)
            for sy in s['state_global']['symbols']:
                rwd.append(s['reward'][sy]['r'])
        rwd = np.array(rwd)

        with avg_reward_writer.as_default():
            tf.summary.scalar('avg_reward', rwd.mean(), step=epoch,
                              description=f"Avg. reward of the last {self.retrain_days} states.")
            avg_reward_writer.flush()