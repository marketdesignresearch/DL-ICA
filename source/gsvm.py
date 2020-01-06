#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file acts as a Java-Python "bridge". It enables a restricted usage in Python of some functionalities of the Global Synergy Value Model (GSVM) imported from the Spectrum Auction Test Suite (SATS), which is written in Java.

It consists of a class called _Gsvm that has the following functionalities:
    0.CONSTRUCTOR:  __init__(self, seed, number_of_national_bidders, number_of_regional_bidders)
        seed = is a number for initializing the random seed generator. Note the auction instance is sampled randomly, i.e., eeach bidder's parameters, such as in which item he is interested most, is sampled randomly when creating an instance.
        number_of_national_bidders = defines the number of bidders from the type: national
        number_of_regional_bidders = defines the number of bidders from the type: regional
        The default parametrization is: One national bidder and Five regional bidders
    1.METHOD: get_bidder_ids(self):
        Returns the bidder_ids as dict_keys.
        In the default parametrization the regional bidders have ids:0,1,2,3,4,5 and the national bidders has id:6
    2.METHOD: get_good_ids(self):
        Returns the ids of the goods as dict keys. In GSVM there are 18 items, representing the regions for the spectrum license.
    3.METHOD: calculate_value(self, bidder_id, goods_vector):
        bidder_id = unique id a bidder in the created _Gsvm instance
        goods_vector = indicator vector of the corresponding bundle for which the value should be queried (list or numpy array with dimension 18)
        Returns the (true) value of bidder bidder_id for a bundle of items goods_vector.
    4.METHOD: get_random_bids(self, bidder_id, number_of_bids, seed=None, mean_bundle_size=9, standard_deviation_bundle_size=4.5):
        bidder_id = unique id a bidder in the created _Gsvm instance
        number_of_bids = number of desired random bids
        seed = initializing the random generator for the random bids
        mean_bundle_size = mean of normal distribution. Represents the average number of 1's in the bundle vector.
        standard_deviation_bundle_size = standard deviation of normal distribution.
        This returns a list of lists of bundle-value pairs, which are sampled randomly accroding to the following procedure:
            First sample a normal random variable Z with parameters mean_bundle_size and standard_deviation_bundle_size.
            Then sample uniformly at random from all bundles in the space that contain excatly Z 1's
            (Note this sampling scheme is different from sampling uniformly at random from the bundle space. It has heavier tails, thus one obtains also samples from bundles with few and many 1's.)
    5.METHOD: get_efficient_allocation(self):
        Returns the efficient, i.e., optimal, allocation (as dict) and the corresponding social welfare (float) of the _Gsvm instance.

This class should not be called directly. Instead it should be used only via the class pysats.py. See example_javabridge.py for an example of how to use the class  _Gsvm.
"""

# Libs
from jnius import JavaClass, MetaJavaClass, JavaMethod, cast, autoclass

__author__ = 'Fabio Isler, Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'

# %%
SizeBasedUniqueRandomXOR = autoclass(
    'org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR')
JavaUtilRNGSupplier = autoclass(
    'org.spectrumauctions.sats.core.util.random.JavaUtilRNGSupplier')
Bundle = autoclass(
    'org.spectrumauctions.sats.core.model.Bundle')
GSVMStandardMIP = autoclass(
    'org.spectrumauctions.sats.opt.model.gsvm.GSVMStandardMIP')

class _Gsvm(JavaClass, metaclass=MetaJavaClass):
    __javaclass__ = 'org/spectrumauctions/sats/core/model/gsvm/GlobalSynergyValueModel'

    # TODO: I don't find a way to have the more direct accessors of the DefaultModel class. So for now, I'm mirroring the accessors
    #createNewPopulation = JavaMultipleMethod([
    #    '()Ljava/util/List;',
    #    '(J)Ljava/util/List;'])
    setNumberOfNationalBidders = JavaMethod('(I)V')
    setNumberOfRegionalBidders = JavaMethod('(I)V')
    createWorld = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Lorg/spectrumauctions/sats/core/model/gsvm/GSVMWorld;')
    createPopulation = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/model/World;Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Ljava/util/List;')

    population = {}
    goods = {}
    efficient_allocation = None

    def __init__(self, seed, number_of_national_bidders, number_of_regional_bidders):
        super().__init__()
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()

        self.setNumberOfNationalBidders(number_of_national_bidders)
        self.setNumberOfRegionalBidders(number_of_regional_bidders)

        world = self.createWorld(rng)
        self._bidder_list = self.createPopulation(world, rng)

        # Store bidders
        bidderator = self._bidder_list.iterator()
        while bidderator.hasNext():
            bidder = bidderator.next()
            self.population[bidder.getId()] = bidder

        # Store goods
        goods_iterator = self._bidder_list.iterator().next().getWorld().getLicenses().iterator()
        count = 0
        while goods_iterator.hasNext():
            good = goods_iterator.next()
            assert good.getId() == count
            count += 1
            self.goods[good.getId()] = good


    def get_bidder_ids(self):
        return self.population.keys()

    def get_good_ids(self):
        return self.goods.keys()

    def calculate_value(self, bidder_id, goods_vector):
        assert len(goods_vector) == len(self.goods.keys())
        bidder = self.population[bidder_id]
        bundle = Bundle()
        for i in range(len(goods_vector)):
            if goods_vector[i] == 1:
                bundle.add(self.goods[i])
        return bidder.calculateValue(bundle).doubleValue()

    def get_random_bids(self, bidder_id, number_of_bids, seed=None, mean_bundle_size=9, standard_deviation_bundle_size=4.5):
        bidder = self.population[bidder_id]
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()
        valueFunction = cast('org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR',
                                bidder.getValueFunction(SizeBasedUniqueRandomXOR, rng))
        valueFunction.setDistribution(
            mean_bundle_size, standard_deviation_bundle_size)
        valueFunction.setIterations(number_of_bids)
        xorBidIterator = valueFunction.iterator()
        bids = []
        while (xorBidIterator.hasNext()):
            xorBid = xorBidIterator.next()
            bid = []
            for good_id, good in self.goods.items():
                if (xorBid.getLicenses().contains(good)):
                    bid.append(1)
                else:
                    bid.append(0)
            bid.append(xorBid.value)
            bids.append(bid)
        return bids

    def get_efficient_allocation(self):
        if self.efficient_allocation:
            return self.efficient_allocation, sum([self.efficient_allocation[bidder_id]['value'] for bidder_id in self.efficient_allocation.keys()])

        mip = GSVMStandardMIP(self._bidder_list)
        mip.setDisplayOutput(True)

        item_allocation = cast('org.spectrumauctions.sats.opt.domain.ItemAllocation', mip.calculateAllocation())

        self.efficient_allocation = {}

        for bidder_id, bidder in self.population.items():
            self.efficient_allocation[bidder_id] = {}
            self.efficient_allocation[bidder_id]['good_ids'] = []
            bidder_allocation = item_allocation.getAllocation(bidder)
            good_iterator = bidder_allocation.iterator()
            while good_iterator.hasNext():
                self.efficient_allocation[bidder_id]['good_ids'].append(good_iterator.next().getId())

            self.efficient_allocation[bidder_id]['value'] = item_allocation.getTradeValue(bidder).doubleValue()

        return self.efficient_allocation, item_allocation.totalValue.doubleValue()
