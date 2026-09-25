using System.Collections.Generic;
using IrisNeuralNet.Diagnostics;
using Xunit;

namespace IrisNeuralNet.Tests;

public sealed class MetricsRingTests
{
    [Fact]
    public void CopyTo_PreservesOrderAndCount()
    {
        var ring = new MetricsRing(capacity: 8);
        for (int epoch = 1; epoch <= 5; epoch++)
        {
            ring.OnEpochCompleted(new EpochMetrics(epoch, 0f, 0f, 0f, 0f));
        }

        var snapshot = new List<EpochMetrics>();
        ring.CopyTo(snapshot);

        Assert.Equal(5, snapshot.Count);
        Assert.Equal(1, snapshot[0].Epoch);
        Assert.Equal(5, snapshot[4].Epoch);
    }

    [Fact]
    public void Wrap_KeepsLatestItems()
    {
        var ring = new MetricsRing(capacity: 4);
        for (int epoch = 1; epoch <= 6; epoch++)
        {
            ring.OnEpochCompleted(new EpochMetrics(epoch, 0f, 0f, 0f, 0f));
        }

        var snapshot = new List<EpochMetrics>();
        ring.CopyTo(snapshot);

        Assert.Equal(4, snapshot.Count);
        Assert.Equal(3, snapshot[0].Epoch);
        Assert.Equal(6, snapshot[3].Epoch);
    }

    [Fact]
    public void Clear_ResetsRing()
    {
        var ring = new MetricsRing(capacity: 4);
        ring.OnEpochCompleted(new EpochMetrics(1, 0f, 0f, 0f, 0f));

        ring.Clear();

        Assert.Equal(0, ring.Count);
    }
}