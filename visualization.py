import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pylab

class VisualizationTool:
    def __init__(self, data):
        self.data = data

    def get_feature_distribution(self, feature):
        '''Displays the distribution of the feature
        
        Args:
        data(pd.Series) = feature for histogram
        '''
        plt.hist(self.data[feature], facecolor='g')
        plt.title('Histogram')
        plt.grid(True)
        plt.show()

    def compare_numerical_features(self, features, colour):
        '''Returns feature distributions and their dependency

        Args:
        data(pd.DataFrame) = data for checking
        features(List[str]) = featurws for comparison
        colour(sns.palettes._ColorPalette) = colour for graphs
        '''
        sns.pairplot(self.data[features], hue='Exited', palette=colour)

    def compare_numerical_and_categorical_features(self, cat_feature, num_feature, 
                                               colour):
        '''Illustrates the bond between numerical and categorical

        Args:
        data(pd.DataFrame) =  data for checking
        cat_features(str) = categorical features for visualization
        num_features(int) = numerical features for visualization
        colour(sns.palettes._ColorPalette) = colour for graphs
        '''
        fig, ax1 = plt.subplots(ncols=1, figsize=(16, 6))
        s = sns.stripplot(ax=ax1, x=cat_feature, y=num_feature, data=self.data,
                        hue='Exited', palette=colour, dodge=True)
        
        plt.show()

    def get_stat_category_features(self, feature, colour,
                               horizontal_located):
        '''Returns three histograms for one categorical feature
        Illustrates statistics on the number of observations for each category, 
        Distribution of the target by groups, 
        how many people did not repay the loan in each category

        Args:
        data(pd.DataFrame) = data for checking
        features(List[str]) = feature for visualization
        colour(sns.palettes._ColorPalette) = colour for graphs
        horizontal_located(bool) = should the graphs be horizontal
        '''
        if horizontal_located:
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 6))
            plt.subplots_adjust(wspace=0.5)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 16))
            plt.subplots_adjust(hspace=0.9)
        
        
        s1 = sns.countplot(ax=ax1, x=feature, data=self.data, palette=colour)
        s1.set(xlabel=feature)
        s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
        s1.set_title('Количество объектов в каждой категории')

        
        group = self.data.groupby(['Exited', feature], as_index=False).count()
        group = group[group.Exited == 1]
        non_returnees = group.sum().RowNumber
        group['persent_non_returnees'] = group.RowNumber / (non_returnees / 100)

        s2 = sns.barplot(ax=ax2, x=feature, y='persent_non_returnees', data=group, palette=colour)
        s2.set(xlabel=feature)
        s2.set_xticklabels(s2.get_xticklabels(),rotation=90)
        s2.set_title('Распределение таргета по категориям')
        
        group = self.data.groupby([feature], as_index=False).count()
        group_target = self.data.groupby(['Exited',feature], as_index=False).count()
        group_target = group_target[group_target.Exited == 1]
        group = group.loc[group[feature].isin(list(group_target[feature].unique()))]
        group_target.index = np.arange(len(group_target))
        group.index = np.arange(len(group))
        group_target['persent'] = (group_target.RowNumber / (group.RowNumber / 100))
        
        s3 = sns.barplot(ax=ax3, x=feature, y='persent', data=group_target, palette=colour)
        s3.set(xlabel=feature)
        s3.set_xticklabels(s3.get_xticklabels(),rotation=90)
        s3.set_title('Процент клиентов ушедших из банка для каждой категории')
            
        plt.show()

    def get_dynamics_numerical_features(self, feature, number_of_bins):
        '''Displays the change of numerical features
        Splits a numerical feature into bins and calculates the average of the target for each bins
        Counts the number of objects for each bin
        
        Agrs:
        data(pd.DataFrame) = data for checking
        feature(str) = feature for visualization
        number_of_bins(int) = the number of bins to divide the feature 
        '''
        self.data.interval = pd.cut(self.data[feature], bins=number_of_bins, right=False).sort_values(ascending=True).astype(str)
        target = [self.data[self.data.interval == trh].Exited.mean() for trh in self.data.interval.unique()]
        count = [self.data[self.data.interval == trh].shape[0] for trh in self.data.interval.unique()]
        
        fig = plt.figure(figsize=(12.,4.))
        plt.subplot(1, 2, 1)
        plt.xlabel(feature)
        plt.xticks(rotation=90)
        plt.title('Среднее значение таргета в каждом бакете')
        plt.grid(True)
        plt.plot(self.data.interval.unique(), target)
        
        plt.subplot(1, 2, 2)
        plt.bar(self.data.interval.unique(), count)  
        plt.xlabel(feature)
        plt.xticks(rotation=90)
        plt.title('Количество объектов в каждом бакете')
        plt.grid(True)
        
        plt.subplots_adjust(wspace=0.3)
        plt.show()
