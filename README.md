# UnityJobFFT
A C# FFT implementation that uses Unity's Job system.

Simply import the file ```FFT.cs``` into your Unity 2018+ project, which will need to target .NET 4.0.

Example use:

```
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;
using Unity.Collections;

using Complex = System.Numerics.Complex;

public class FFTExample : MonoBehaviour
{
	public FFT.ESize fftSize = FFT.ESize.N16;

	private FFT fft;
	private List<JobHandle> transformJobs = new List<JobHandle>();
  
  	private Complex[] someComplexData;

	private void Awake() {
		fft = new FFT(fftSize);
	}

	private void Update() {
		if(transformJobs.Count > 0)
			return;
      
    		//Schedule some FFT jobs with your data. Data is transformed inline.
		transformJobs.Add(fft.ScheduleTransform(ref someComplexData));

		JobHandle.ScheduleBatchedJobs();
	}

	private void LateUpdate() {
    		//Wait for the jobs to complete.
		foreach(JobHandle job in transformJobs) {
			if(!job.IsCompleted)
				return;
		}
    
    		//Ensure jobs are completed.
		foreach(JobHandle job in transformJobs) {
			job.Complete();
		}
    
    		//someComplexData has now been transformed, and may be used.

		transformJobs.Clear();
	}

	private void OnDestroy() {
		foreach(JobHandle job in transformJobs) {
			job.Complete();
		}

		transformJobs.Clear();
		fft.Dispose();
	}
}
```

