# Current simulator harness

Sadhana runs two methods in the native ABC YAM task. Both methods use the original author learner. Each method has a separate installed Python environment. The adapter connects that learner to the simulator.

The root agent owns GPU0 admission and execution. It checks the actual child exit codes and result files. It keeps the other GPUs available to their owners. It does not start a second GPU0 job while the lease is active.

The FastTD3 agent owns the state and action adapter. The reward agent defines the task reward and its geometry. FastTD3 starts with a random state policy. It learns two critics from environment rewards. The new reward uses current full success and bounded geometric progress. The original task judge still measures five bottles in the bin.

The EXPO agent owns the image-policy adapter. Task adaptation uses three complete TRAIN demonstrations. A separate held-out split selects the adapted actor. Ordinary EXPO then uses its original learner and ten completed online warmup episodes. Its first update trigger makes three learner calls. The live camera and prefetch memory check must pass before that long run starts.

ABC owns the physics, cameras, robot model and task judge. The state host runs 128 CPU worlds for FastTD3. The image host uses the original cameras for pi0.5. Each host uses the same physical action order and records its declared action mapping.

Frozen evaluation restores the declared scene and complete physics state. The initial camera frames must match before actions start. The policy does not train during this evaluation. Results include the native task score and the actual video.

The viewer shows measured task scores. It labels live training counts separately. A completed capacity update is a memory result. It is not evidence of task success.

These methods have different starting policies, observations and rewards. Their scores compare training strategies in this task. They do not isolate an algorithm advantage. Both the reward and the training budget change in the new FastTD3 experiment.

Atlas and Engram own source data and annotations. Leela owns the calibrated R1 Lite scene and control contract. That training release is still pending. The dashed R1 Lite connection in the diagram is planned. The current task results use YAM.

This explanation uses ASD-STE100 guidance. A formal compliance check has not been done. This is a current architecture draft. Goal completion and final benchmark acceptance remain pending.

See [diagram source](ORIGINAL_METHODS_ARCHITECTURE.mmd). Solid arrows show data flow. Dashed arrows show control flow or a connection marked as planned. The diagram shows implemented software; a connection does not establish native test acceptance.
