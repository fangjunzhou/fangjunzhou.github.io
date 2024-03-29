---
layout: post
title: Dive into ECS 1 - Entities
category: miscellaneous
---

# ECS Overview

As many of you may know, entities are a very important part of the ECS Framework. However, I used to learn that entities are created contiguously in the memory. But I had no idea how they are allocated in practice. After some research, I learned some more underlying working principles of ECS.

First, entities are populated by archetypes inside the memory. These archetypes not only describe the feature of entities but also group their data to increase the cache hit rate, which is the ultimate goal of the ECS framework.

Inside archetypes, chucks form the underlying data structure of the archetype. Simply put, chunks are linked list **nodes** or array list **elements** that point to the next chunk and contain contiguous entities data. Note that here I use linked list nodes (array list elements) instead of a linked list (array list). This is because the chunk itself uses contiguous memory to store entities data. But chunks are stored in another data structure for archetype to search.

So why not use an entire array to store all the data inside one archetype? This is also the brightest decision. While entities may be instantiated and destroyed from time to time. Using a long array to manage all the entities may lead to gaps formed by destroyed entities. Using chunks to manage entities gives developers (I mean in Unity) more freedom to manage arrays individually. Which lower cache miss in return. I'll talk about that later.

# Archetype

Let's start with the Archetype. This is the basis of entities classification.

On Entities manual page, Unity mentioned[^1] that

> As you create entities and add components to them, the EntityManager keeps track of the unique combinations of components on the existing entities. Such a unique combination is called an Archetype.
> An EntityArchetype is a unique combination of component types. The EntityManager uses the archetype to group all entities that have the same sets of components.

This may mean that archetypes contain all the entities with a certain combination of components. It also means that every time you add a new component, EntityManager may try to figure out if you create a new component combination or if the entity you created can fall into some existed archetype.

One point I want to mention is that I used to think that one entity can belong to different archetypes.

For example, if an entity `E1` has components `A`, `B`, and `C`, it may fall into the archetype `[A]`, `[B]`, `[C]`, `[A, B]`, `[A, C]`, `[B, C]`, `[A, B, C]`.

Why I had this idea because I thought if there's another entity `E2` with components `A`, `B`, and `D`, they will both fall into the archetype `[A, B]`. It sounds easier to search entities with components `A` and `B` (This is also one of the operations that can only be done efficiently and easily in ECS).

However, it's **completely wrong**!!

In reality, there will only be archetype `[A, B, C]` containing `E1` and archetype `[A, B, D]` containing `E2`.

Then how can ECS perform the search as I mentioned so fast? The answer is simple: the system can find all the archetypes containing **`A` and `B`**, which are archetype `[A, B, C]` and archetype `[A, B, D]` here. Then we only need to loop through all the objects in these archetypes.

This graph[^2] from ECS Manual shows the process I mentioned pretty clear:

![picture 1](/images/2022-04-08-01-00-33-archetype-foreach.png)

It's worth mentioning that entities can be created through archetypes. This means the created entities will have a certain combination of components.

---

{: data-content="Following content needs further research"}

Creating entities through archetypes can be useful in scenarios like enemy spawning. Since all the enemies will have the same components (Health, Movement, Weapon, Navigation for example), spawning with archetype may be faster. Since you don't need to create an entity and add different components to it.

Spawning with archetypes may also avoid creating unnecessary archetypes. This is because unwanted archetypes may be created in the process of adding components.

---

# Creating Entities

Entities can be created in four ways:

- Create an entity with components that use an array of ComponentType objects.
- Create an entity with components that use an EntityArchetype.
- Copy an existing entity, including its current data, with Instantiate
- Create an entity with no components and then add components to it. (You can add components immediately or when additional components are needed.)

# Chunks

The DOTS ECS uses chunks to store entities in each archetype. As I mentioned, this is a brilliant idea.

First, we can observe the chunk structure[^3]:

![picture 2](/images/2022-04-08-01-01-34-chunk-structure.png)

It can be seen that each archetype is composed of several chunks. For archetypes, new chunks should be able to be created and added to the existed chunk list. But for chunks, the number of entities inside should be fixed.

This can be proved in the ECS Scripting API for EntityArchetype[^4]:

> | Name          | Description                                                                              |
> | :------------ | :--------------------------------------------------------------------------------------- |
> | ChunkCapacity | The number of entities having this archetype that can fit into a single chunk of memory. |
> | ChunkCount    | The current number of chunks storing entities having this archetype.                     |

Here, the `ChunkCapacity` describes how many entities can each chunk stores. While the `ChunkCount` is the number of chunks for each archetype.

Interestingly, in the documentation for `ChunkCapacity` [^5], Unity states that

> Capacity is determined by the fixed, 16KB size of the memory blocks allocated by the ECS framework and the total storage size of all the component types in the archetype.

This means that although the `ChunkCapacity` is read-only, it is not a constant. Instead, the physical chunk size in memory is fixed (16KB here). Thus, `ChunkCapacity` will be smaller for larger entities. It can also be deducted that smaller entities may have better performance in loops.

---

{: data-content="footnotes"}

[^1]: [Unity Entities Manual](https://docs.unity3d.com/Packages/com.unity.entities@0.50/manual/ecs_entities.html)
[^2]: [ECS Core Introduction](https://docs.unity3d.com/Packages/com.unity.entities@0.50/manual/ecs_core.html)
[^3]: [ECS Core Introduction](https://docs.unity3d.com/Packages/com.unity.entities@0.50/manual/ecs_core.html)
[^4]: [ECS Scripting API - Struct EntityArchetype](https://docs.unity3d.com/Packages/com.unity.entities@0.50/api/Unity.Entities.EntityArchetype.html)
[^5]: [ECS Scripting API - Property ChunkCapacity](https://docs.unity3d.com/Packages/com.unity.entities@0.50/api/Unity.Entities.EntityArchetype.ChunkCapacity.html#Unity_Entities_EntityArchetype_ChunkCapacity)
