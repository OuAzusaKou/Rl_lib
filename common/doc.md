[TOC]

#### Agent
- ##### Actor;
	- a = pi(s) [DDPG]
	- Distribution(a) = pi(s) [SAC]
- ##### Critic；
	- Q（s,a） [DDPG，SAC]
	- Q(s)[a]  [DQN]
- ##### Optimizer
	- Adam
	- SGD
- ##### Dicriminator
	- p = D(s,a)

#### Algorithms
- ##### Agent
- ##### Critic update(agent)
	- ###### DDPG_SAC_DQN critic update
			input: St,At,Rt,St+1,At+1，Agent
			y = Rt + Q(St+1,pi(St+1))
			loss = MSE(y,Q(s,a))
			critic.optimizer(loss)
	- ###### GAIL critic update
		- ****
- ##### Actor update(agent)
	- ###### DDPG Actor update
			 input: S,A,R,S,A,Agent
			 loss =  - Q(s,pi(s))
			 actor.optimizer(loss)
	- ###### SAC Actor update
	- ###### GAIL Actor update
- ##### DataCollection
- ##### Allocation(***)


#### DataCollection
- Memory;
- add;
- extract;

#### Editor.md directory

    editor.md/
            lib/
            css/
            scss/
            tests/
            fonts/
            images/
            plugins/
            examples/
            languages/     
            editormd.js
            ...

```html
<!-- English -->
<script src="../dist/js/languages/en.js"></script>

<!-- 繁體中文 -->
<script src="../dist/js/languages/zh-tw.js"></script>
```