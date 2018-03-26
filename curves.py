def learning_curve(x_train,y_train,algorithm = None):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        quantity = range(40000,880000,40000)
    
        train_scores = [algorithm.fit(x_train[:quty],y_train[:quty]).score(x_train[:quty],y_train[:quty]) for quty in quantity]
    
        test_scores = [algorithm.fit(x_train[:quty],y_train[:quty]).score(x_train[-quty:],y_train[-quty:]) for quty in quantity]
    
        plt.plot(quantity, train_scores, "o-", color="blue", label="train")
        plt.plot(quantity, test_scores, "o-", color="red", label="test")
        plt.grid(True);
        plt.xlabel("Cantidad de datos")
        plt.ylabel("score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

def progress_curve(x_train,y_train):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from sklearn.model_selection import train_test_split, cross_val_score, KFold
        from sklearn.ensemble import RandomForestRegressor
        
        max_depths = range(1,21)
        r = np.array([(np.mean(j), np.std(j)) for j in [cross_val_score(RandomForestRegressor(max_depth=i,random_state=2),x_train[:],y_train[:],cv=KFold(len(x_train[:]), 5)) for i in max_depths]])
        plt.plot(max_depths, r[:,0], "o-", color="blue", label="test")
        plt.fill_between(max_depths, r[:,0]-r[:,1], r[:,0]+r[:,1], color="blue", alpha=.2)
        plt.xlabel("Profundidad del RandomForestRegressor.")
        plt.ylabel("accuracy")
        plt.grid(True);
        blue_line = mlines.Line2D([], [], color='blue',label='test')
        plt.legend(handles=[blue_line])

        plt.show()
