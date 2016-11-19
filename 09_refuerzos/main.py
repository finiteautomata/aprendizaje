#! coding: utf-8
"""Grid world example."""
import numpy as np
import random
from collections import defaultdict
import pylab


class GridWorld(object):
    """Clase que implementa un gridWorld"""

    def __init__(self, width=3, height=2, goals=None, positive=100,
                 select_action='random', alpha=0.9, gamma=0.9):
        """Seteo los parametros"""
        self.height = height
        self.width = width
        self.positive = positive
        self.select_action = select_action
        self.alpha = alpha
        self.gamma = gamma
        self.learning_step = 0

        """Represento Q como una lista de filas de celdas de Q
        Es decir, en self.Q[0][1] esta la celda que corresponde al estado
        [0,1]. La celdas son diccionarios de acciones posibles y sus
        valores.

        Por ejemplo, self.Q[0][0] podria ser: {'right':0.8, 'down':0.3'}
        o si el estado [0,2] es terminal, self.Q[0][2]: {'stay':100}

        Esta definido con un defaultdict para facilidad, podria
        inicialiazarse explicitamente
        """

        self.Q = [[defaultdict(lambda:random.random()) for _ in
                   range(self.width)] for _ in range(self.height)]

        if goals is None:
            self.goals = [[0, self.width - 1]]
        else:
            self.goals = list(goals)

    def possible_actions(self, state):
        """Devuelve acciones posibles.

        Tiene en cuenta bordes y eso
        """
        row, col = state
        res = []
        if row > 0:
            res.append('up')
        if row < (self.height - 1):
            res.append('down')
        if col > 0:
            res.append('left')
        if col < (self.width - 1):
            res.append('right')

        return res

    def move(self, state, action):
        """Dado un estado y una accion devuelve un nuevo estado."""
        new_state = list(state)
        if action == 'up':
            new_state[0] -= 1
        if action == 'down':
            new_state[0] += 1
        if action == 'right':
            new_state[1] += 1
        if action == 'left':
            new_state[1] -= 1

        return new_state

    def is_terminal(self, state):
        """Me dice si state es terminal"""
        return state in self.goals

    def reward(self, state, action):
        u"""Función de recompensa."""
        if self.is_terminal(self.move(state, action)):
            return self.positive
        return 0

    def q_value(self, state, action):
        """Devuelve valor de Q."""
        return self.Q[state[0]][state[1]][action]

    def new_q_value(self, state, action):
        u"""Devuelve el nuevo valor de Q."""
        a = self.alpha
        r = self.reward(state, action)

        old_q = self.q_value(state, action)

        next_state = self.move(state, action)
        next_actions = self.possible_actions(next_state)
        next_best_q = max(self.q_value(next_state, a) for a in next_actions)

        return (1 - a) * old_q + a * (r + self.gamma * next_best_q)

    def set_q(self, state, action, value):
        """Setea Q a nuevo valor."""
        self.Q[state[0]][state[1]][action] = value

    def learn(self, state):
        """Funcion que aprende, implementa Qlearning."""
        self.learning_step += 1

        print("Estado inicial: {}".format(state))
        # Repito hasta que state sea terminal
        while not self.is_terminal(state):
            action = random.choice(self.possible_actions(state))
            new_state = self.move(state, action)
            # 3) Calculo el nuevo valor de Q(s,a)

            self.set_q(state, action, self.new_q_value(state, action))

            # 4) Actualizo s
            print("{} ---> {}".format(state, new_state))
            state = new_state

    def draw(self):
        """Funcion para plotear Q."""
        def if_except_return_nan(dic, k):
            try:
                return dict(dic)[k]
            except:
                return np.nan

        matrix_right = np.array([[if_except_return_nan(cel,'right') for cel in row] for row in self.Q])
        matrix_left = np.array([[if_except_return_nan(cel,'left') for cel in row] for row in self.Q])
        matrix_up = np.array([[if_except_return_nan(cel,'up') for cel in row] for row in self.Q])
        matrix_down = np.array([[if_except_return_nan(cel,'down') for cel in row] for row in self.Q])
        matrix_stay = np.array([[if_except_return_nan(cel,'stay') for cel in row] for row in self.Q])

        fig = pylab.figure(figsize=2*np.array([self.width,self.height]))
        for i in range(matrix_right.shape[0]):
            for j in range(matrix_right.shape[1]):
                if not np.isnan(matrix_stay[i][j]): pylab.text(j-.5, self.height- i-.5,'X')
                else:
                    pylab.text(j + 0.2 - .5, self.height - i - .5, str(matrix_right[i][j])[0:4]+">")
                    pylab.text(j-0.3-.5, self.height-i-.5,"<"+str(matrix_left[i][j])[0:4])

                    pylab.text(j-.5, self.height- i+.1-.5,str(matrix_up[i][j])[0:4])
                    pylab.text(j-.5, self.height- i-.1-.5,str(matrix_down[i][j])[0:4])

        pylab.xlim(-1,self.width-1)
        pylab.ylim(0,self.height)
        pylab.xticks(range(self.width),map(str,range(self.width)))
        pylab.yticks(range(self.height+1),reversed(map(str,range(self.height+1))))
        pylab.grid(color='r', linestyle='-', linewidth=2)
        pylab.title('Q (learning_step:%d)' % self.learning_step,size=16)
        fig.tight_layout()


if __name__ == "__main__":
    # Ejemplo de 4x4 con goal en el medio
    # gw =gridWorld(height=4,width=4,goals=[[2,2]])


    # Ejemplo de gridWorld  de 2x3
    gw = GridWorld(height=4,width=4,goals=[[2,2]])

    # Entreno 1K veces
    for epoch in range(1000):
        # Ploteo la matrix a los 10,200, y 999 epochs
        print("Episodio {}".format(epoch))
        if epoch == 10: gw.draw()
        if epoch == 200: gw.draw()
        if epoch == 999: gw.draw()

        # Elijo un state random para empezar
        start_state = [ random.randint(0,gw.height-1),random.randint(0,gw.width-1)]

        # Entreno
        gw.learn(start_state)

    pylab.show()
