# UnityJobFFT
A C# FFT implementation that uses Unity's Job system.

Simply import the file ```FFT.cs``` into your Unity 2018+ project, which will need to target .NET 4.0.

Example use:

```
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;
using Unity.Collections;

public class FFTExample : MonoBehaviour
{
	private FFT fft = new FFT(FFT.ESize.N16);
	private List<JobHandle> transformJobs = new List<JobHandle>();
  
  	private Complex[] someComplexData, someComplexData2, someComplexData3;

	private void Update() {
		if(transformJobs.Count > 0)
			return;
      
    		//Schedule multiple FFT jobs with your data. Data is transformed inline.
		transformJobs.Add(fft.ScheduleTransform(ref someComplexData));
		transformJobs.Add(fft.ScheduleTransform(ref someComplexData2));
		transformJobs.Add(fft.ScheduleTransform(ref someComplexData3));

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
    
    		//someComplexData has now been transformed, and may be used!

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

