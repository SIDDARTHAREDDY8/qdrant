use std::collections::HashSet;
use std::time::Duration;

use collection::collection::Collection;
use collection::collection::distance_matrix::{
    CollectionSearchMatrixRequest, CollectionSearchMatrixResponse,
};
use collection::config::ShardingMethod;
use collection::grouping::GroupBy;
use collection::grouping::group_by::{GroupRequest, SourceRequest};
use collection::operations::consistency_params::ReadConsistency;
use collection::operations::point_ops::WriteOrdering;
use collection::operations::routing::RoutingToken;
use collection::operations::shard_selector_internal::ShardSelectorInternal;
use collection::operations::types::*;
use collection::operations::universal_query::collection_query::{
    CollectionPrefetch, CollectionQueryRequest,
};
use collection::operations::{CollectionUpdateOperations, OperationWithClockTag};
use collection::shards::shard_trait::WaitUntil;
use collection::{discovery, recommendations};
use common::counter::hardware_accumulator::HwMeasurementAcc;
use futures::TryStreamExt as _;
use futures::stream::FuturesUnordered;
use segment::data_types::facets::{FacetParams, FacetResponse};
use segment::types::{PointIdType, ScoredPoint, ShardKey, WithPayloadInterface, WithVector};
use shard::retrieve::record_internal::RecordInternal;
use shard::scroll::ScrollRequestInternal;
use shard::search::CoreSearchRequestBatch;

use super::TableOfContent;
use crate::content_manager::errors::{StorageError, StorageResult};
use crate::rbac::Auth;

impl TableOfContent {
    /// Recommend points using positive and negative example from the request
    ///
    /// # Arguments
    ///
    /// * `collection_name` - for what collection do we recommend
    /// * `request` - [`RecommendRequestInternal`]
    ///
    /// # Result
    ///
    /// Points with recommendation score
    #[allow(clippy::too_many_arguments)]
    pub async fn recommend(
        &self,
        collection_name: &str,
        request: RecommendRequestInternal,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        shard_selector: ShardSelectorInternal,
        auth: Auth,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<Vec<ScoredPoint>> {
        let collection_pass = auth.check_point_op(collection_name, &request, "recommend")?;

        let collection = self.get_collection(&collection_pass).await?;
        self.validate_recommend_lookup_from(&request).await?;
        recommendations::recommend_by(
            request,
            &collection,
            |name| self.get_collection_opt(name),
            read_consistency,
            routing_token,
            shard_selector,
            timeout,
            hw_measurement_acc,
        )
        .await
        .map_err(|err| err.into())
    }

    /// Recommend points in a batching fashion using positive and negative example from the request
    ///
    /// # Arguments
    ///
    /// * `collection_name` - for what collection do we recommend
    /// * `requests` - [`RecommendRequestBatch`]
    ///
    /// # Result
    ///
    /// Points with recommendation score
    #[allow(clippy::too_many_arguments)]
    pub async fn recommend_batch(
        &self,
        collection_name: &str,
        mut requests: Vec<(RecommendRequestInternal, ShardSelectorInternal)>,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        auth: Auth,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<Vec<Vec<ScoredPoint>>> {
        let mut collection_pass = None;
        for (request, _shard_selector) in &mut requests {
            collection_pass =
                Some(auth.check_point_op(collection_name, request, "recommend_batch")?);
        }
        let Some(collection_pass) = collection_pass else {
            return Ok(vec![]);
        };

        let collection = self.get_collection(&collection_pass).await?;
        for (request, _shard_selector) in &requests {
            self.validate_recommend_lookup_from(request).await?;
        }
        recommendations::recommend_batch_by(
            requests,
            &collection,
            |name| self.get_collection_opt(name),
            read_consistency,
            routing_token,
            timeout,
            hw_measurement_acc,
        )
        .await
        .map_err(|err| err.into())
    }

    /// Search in a batching fashion for the closest points using vector similarity with given restrictions defined
    /// in the request
    ///
    /// # Arguments
    ///
    /// * `collection_name` - in what collection do we search
    /// * `request` - [`CoreSearchRequestBatch`]
    /// * `shard_selection` - which local shard to use
    /// * `timeout` - how long to wait for the response
    /// * `read_consistency` - consistency level
    ///
    /// # Result
    ///
    /// Points with search score
    #[allow(clippy::too_many_arguments)]
    pub async fn core_search_batch(
        &self,
        collection_name: &str,
        mut request: CoreSearchRequestBatch,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        shard_selection: ShardSelectorInternal,
        auth: Auth,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<Vec<Vec<ScoredPoint>>> {
        let mut collection_pass = None;
        for request in &mut request.searches {
            collection_pass =
                Some(auth.check_point_op(collection_name, request, "core_search_batch")?);
        }
        let Some(collection_pass) = collection_pass else {
            return Ok(vec![]);
        };

        let collection = self.get_collection(&collection_pass).await?;
        collection
            .core_search_batch(
                request,
                read_consistency,
                routing_token,
                shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await
            .map_err(|err| err.into())
    }

    /// Count points in the collection.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - in what collection do we count
    /// * `request` - [`CountRequestInternal`]
    /// * `shard_selection` - which local shard to use
    ///
    /// # Result
    ///
    /// Number of points in the collection.
    ///
    #[allow(clippy::too_many_arguments)]
    pub async fn count(
        &self,
        collection_name: &str,
        request: CountRequestInternal,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        timeout: Option<Duration>,
        shard_selection: ShardSelectorInternal,
        auth: Auth,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<CountResult> {
        let collection_pass = auth.check_point_op(collection_name, &request, "count")?;

        let collection = self.get_collection(&collection_pass).await?;
        collection
            .count(
                request,
                read_consistency,
                routing_token,
                &shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await
            .map_err(|err| err.into())
    }

    /// Return specific points by IDs
    ///
    /// # Arguments
    ///
    /// * `collection_name` - select from this collection
    /// * `request` - [`PointRequestInternal`]
    /// * `shard_selection` - which local shard to use
    ///
    /// # Result
    ///
    /// List of points with specified information included
    #[allow(clippy::too_many_arguments)]
    pub async fn retrieve(
        &self,
        collection_name: &str,
        request: PointRequestInternal,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        timeout: Option<Duration>,
        shard_selection: ShardSelectorInternal,
        auth: Auth,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<Vec<RecordInternal>> {
        let collection_pass = auth.check_point_op(collection_name, &request, "retrieve")?;

        let collection = self.get_collection(&collection_pass).await?;
        collection
            .retrieve(
                request,
                read_consistency,
                routing_token,
                &shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await
            .map_err(|err| err.into())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn group(
        &self,
        collection_name: &str,
        request: GroupRequest,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        shard_selection: ShardSelectorInternal,
        auth: Auth,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<GroupsResult> {
        let collection_pass = auth.check_point_op(collection_name, &request, "group")?;

        let collection = self.get_collection(&collection_pass).await?;
        self.validate_group_lookup_from(&request).await?;

        let collection_by_name = |name| self.get_collection_opt(name);

        let group_by = GroupBy::new(request, &collection, collection_by_name, hw_measurement_acc)
            .set_read_consistency(read_consistency)
            .set_routing_token(routing_token)
            .set_shard_selection(shard_selection)
            .set_timeout(timeout);

        group_by
            .execute()
            .await
            .map(|groups| GroupsResult { groups })
            .map_err(|err| err.into())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn discover(
        &self,
        collection_name: &str,
        request: DiscoverRequestInternal,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        shard_selector: ShardSelectorInternal,
        auth: Auth,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<Vec<ScoredPoint>> {
        let collection_pass = auth.check_point_op(collection_name, &request, "discover")?;

        let collection = self.get_collection(&collection_pass).await?;
        discovery::discover(
            request,
            &collection,
            |name| self.get_collection_opt(name),
            read_consistency,
            routing_token,
            shard_selector,
            timeout,
            hw_measurement_acc,
        )
        .await
        .map_err(|err| err.into())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn discover_batch(
        &self,
        collection_name: &str,
        mut requests: Vec<(DiscoverRequestInternal, ShardSelectorInternal)>,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        auth: Auth,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<Vec<Vec<ScoredPoint>>> {
        let mut collection_pass = None;
        for (request, _shard_selector) in &mut requests {
            collection_pass =
                Some(auth.check_point_op(collection_name, request, "discover_batch")?);
        }
        let Some(collection_pass) = collection_pass else {
            return Ok(vec![]);
        };

        let collection = self.get_collection(&collection_pass).await?;

        discovery::discover_batch(
            requests,
            &collection,
            |name| self.get_collection_opt(name),
            read_consistency,
            routing_token,
            timeout,
            hw_measurement_acc,
        )
        .await
        .map_err(|err| err.into())
    }

    /// Paginate over all stored points with given filtering conditions
    ///
    /// # Arguments
    ///
    /// * `collection_name` - which collection to use
    /// * `request` - [`ScrollRequestInternal`]
    /// * `shard_selection` - which local shard to use
    ///
    /// # Result
    ///
    /// List of points with specified information included
    #[allow(clippy::too_many_arguments)]
    pub async fn scroll(
        &self,
        collection_name: &str,
        request: ScrollRequestInternal,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        timeout: Option<Duration>,
        shard_selection: ShardSelectorInternal,
        auth: Auth,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<ScrollResult> {
        let collection_pass = auth.check_point_op(collection_name, &request, "scroll")?;

        let collection = self.get_collection(&collection_pass).await?;
        collection
            .scroll_by(
                request,
                read_consistency,
                routing_token,
                &shard_selection,
                timeout,
                hw_measurement_acc,
            )
            .await
            .map_err(|err| err.into())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn query_batch(
        &self,
        collection_name: &str,
        mut requests: Vec<(CollectionQueryRequest, ShardSelectorInternal)>,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        auth: Auth,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<Vec<Vec<ScoredPoint>>> {
        let mut collection_pass = None;
        for (request, _shard_selector) in &mut requests {
            collection_pass = Some(auth.check_point_op(collection_name, request, "query_batch")?);
        }
        let Some(collection_pass) = collection_pass else {
            // This can happen only if there are no requests
            return Ok(vec![]);
        };

        let collection = self.get_collection(&collection_pass).await?;
        for (request, _shard_selector) in &requests {
            self.validate_query_lookup_from(request).await?;
        }

        collection
            .query_batch(
                requests,
                |name| self.get_collection_opt(name),
                read_consistency,
                routing_token,
                timeout,
                hw_measurement_acc,
            )
            .await
            .map_err(|err| err.into())
    }

    // Return unique values for a payload key, and a count of points for each value.
    #[allow(clippy::too_many_arguments)]
    pub async fn facet(
        &self,
        collection_name: &str,
        request: FacetParams,
        shard_selection: ShardSelectorInternal,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        auth: Auth,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<FacetResponse> {
        let collection_pass = auth.check_point_op(collection_name, &request, "facet")?;

        let collection = self.get_collection(&collection_pass).await?;

        collection
            .facet(
                request,
                shard_selection,
                read_consistency,
                routing_token,
                timeout,
                hw_measurement_acc,
            )
            .await
            .map_err(StorageError::from)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn search_points_matrix(
        &self,
        collection_name: &str,
        request: CollectionSearchMatrixRequest,
        read_consistency: Option<ReadConsistency>,
        routing_token: Option<RoutingToken>,
        shard_selection: ShardSelectorInternal,
        auth: Auth,
        timeout: Option<Duration>,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> Result<CollectionSearchMatrixResponse, StorageError> {
        let collection_pass =
            auth.check_point_op(collection_name, &request, "search_points_matrix")?;

        let collection = self.get_collection(&collection_pass).await?;

        collection
            .search_points_matrix(
                request,
                shard_selection,
                read_consistency,
                routing_token,
                timeout,
                hw_measurement_acc,
            )
            .await
            .map_err(StorageError::from)
    }

    /// Split a point-ID-based update operation into per-shard-key variants, so
    /// that each shard key only receives the point IDs that actually live
    /// under it.
    ///
    /// `ids_per_key` maps every selected shard key to the requested point IDs
    /// that were found under it. Shard keys with no matching IDs are skipped.
    /// Returns `Err(missed_point_id)` for operations that fail on missing
    /// points (payload and vector updates) when a requested ID was not found
    /// under any of the selected shard keys, so the failure is reported
    /// honestly instead of surfacing as a false `not found` after partial
    /// writes.
    fn split_operation_per_shard_key(
        operation: &CollectionUpdateOperations,
        ids_per_key: &[(ShardKey, HashSet<PointIdType>)],
    ) -> Result<Vec<(ShardKey, CollectionUpdateOperations)>, PointIdType> {
        // Not a point-ID-based operation (e.g. filter-based) or one that can
        // create points (e.g. upserts): dispatch the full operation to every
        // shard key, as before.
        let full_dispatch = || {
            ids_per_key
                .iter()
                .map(|(shard_key, _)| (shard_key.clone(), operation.clone()))
                .collect()
        };

        let Some(point_ids) = operation.point_ids() else {
            return Ok(full_dispatch());
        };
        if operation.upsert_point_ids().is_some() || point_ids.is_empty() {
            return Ok(full_dispatch());
        }

        // Payload and vector updates fail on missing points, so a requested ID
        // that lives in none of the selected shard keys is a genuine not
        // found. Deletes of missing points are a no-op and keep the previous
        // behavior instead of surfacing a new error.
        if matches!(
            operation,
            CollectionUpdateOperations::PayloadOperation(_)
                | CollectionUpdateOperations::VectorOperation(_)
        ) {
            let found_point_ids: HashSet<_> = ids_per_key
                .iter()
                .flat_map(|(_, key_point_ids)| key_point_ids.iter().copied())
                .collect();
            if let Some(missed_point_id) = point_ids
                .iter()
                .copied()
                .find(|point_id| !found_point_ids.contains(point_id))
            {
                return Err(missed_point_id);
            }
        }

        let mut operations_per_key = Vec::with_capacity(ids_per_key.len());
        for (shard_key, key_point_ids) in ids_per_key {
            let mut operation_for_key = operation.clone();
            operation_for_key.retain_point_ids(|point_id| key_point_ids.contains(point_id));
            if operation_for_key
                .point_ids()
                .is_some_and(|point_ids| !point_ids.is_empty())
            {
                operations_per_key.push((shard_key.clone(), operation_for_key));
            }
        }
        Ok(operations_per_key)
    }

    /// # Cancel safety
    ///
    /// This method is cancel safe.
    ///
    /// When it is cancelled, the operation may not be applied on some shard keys. But, all nodes
    /// are guaranteed to be consistent.
    async fn _update_shard_keys(
        collection: &Collection,
        shard_keys: Vec<ShardKey>,
        operation: CollectionUpdateOperations,
        wait: WaitUntil,
        timeout: Option<Duration>,
        ordering: WriteOrdering,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<UpdateResult> {
        // `Collection::update_from_client` is cancel safe, so this method is cancel safe.

        // For point-ID-based updates, find out which of the requested point
        // IDs live under each shard key, so that each key only receives its
        // own IDs. Dispatching the full ID list to every key made each replica
        // set report a false `No point with id ... found` for IDs owned by
        // other shard keys, even though the update had already been applied
        // there (see <https://github.com/qdrant/qdrant/issues/10064>).
        let mut operations_per_key = Vec::with_capacity(shard_keys.len());
        let explicit_point_ids = operation.point_ids();
        let can_create_points = operation.upsert_point_ids().is_some();

        if let Some(point_ids) = explicit_point_ids
            && !can_create_points
            && !point_ids.is_empty()
        {
            let lookups: FuturesUnordered<_> = shard_keys
                .iter()
                .cloned()
                .map(|shard_key| {
                    let point_ids = point_ids.clone();
                    let hw_measurement_acc = hw_measurement_acc.clone();
                    async move {
                        let shard_selector = ShardSelectorInternal::ShardKey(shard_key.clone());
                        let records = collection
                            .retrieve(
                                PointRequestInternal {
                                    ids: point_ids,
                                    with_payload: Some(WithPayloadInterface::Bool(false)),
                                    with_vector: WithVector::from(false),
                                },
                                None,
                                None,
                                &shard_selector,
                                timeout,
                                hw_measurement_acc,
                            )
                            .await?;
                        let key_point_ids: HashSet<_> =
                            records.into_iter().map(|record| record.id).collect();
                        StorageResult::Ok((shard_key, key_point_ids))
                    }
                })
                .collect();

            let ids_per_key: Vec<(ShardKey, HashSet<PointIdType>)> = lookups.try_collect().await?;

            match Self::split_operation_per_shard_key(&operation, &ids_per_key) {
                Ok(split) => operations_per_key = split,
                Err(missed_point_id) => {
                    return Err(CollectionError::PointNotFound { missed_point_id }.into());
                }
            }
        }

        // Fall back to the previous behavior when there is nothing to split:
        // non-point-ID operations, operations that can create points, and
        // empty splits (e.g. deletes of already-gone points, which are a
        // no-op).
        if operations_per_key.is_empty() {
            operations_per_key = shard_keys
                .into_iter()
                .map(|shard_key| (shard_key, operation.clone()))
                .collect();
        }

        let updates: FuturesUnordered<_> = operations_per_key
            .into_iter()
            .map(|(shard_key, operation_for_key)| {
                collection.update_from_client(
                    operation_for_key,
                    wait,
                    timeout,
                    ordering,
                    Some(shard_key),
                    hw_measurement_acc.clone(),
                )
            })
            .collect();

        // `Collection::update_from_client` is cancel safe, so it's safe to use `TryStreamExt::try_collect`
        let results: Vec<_> = updates.try_collect().await?;

        results
            .into_iter()
            .next()
            .ok_or_else(|| StorageError::bad_input("Empty shard keys selection"))
    }

    /// # Cancel safety
    ///
    /// This method is cancel safe.
    #[allow(clippy::too_many_arguments)]
    pub async fn update(
        &self,
        collection_name: &str,
        operation: OperationWithClockTag,
        wait: WaitUntil,
        timeout: Option<Duration>,
        ordering: WriteOrdering,
        shard_selector: ShardSelectorInternal,
        auth: Auth,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> StorageResult<UpdateResult> {
        let collection_pass = auth.check_point_op(
            collection_name,
            &operation.operation,
            operation.operation.operation_name(),
        )?;

        // `TableOfContent::_update_shard_keys` and `Collection::update_from_*` are cancel safe,
        // so this method is cancel safe.

        let collection = self.get_collection(&collection_pass).await?;

        // Ordered operation flow:
        //
        // ┌───────────────────┐
        // │ User              │
        // └┬──────────────────┘
        //  │ Shard: None
        //  │ Ordering: Strong
        //  │ ShardKey: Some("cats")
        //  │ ClockTag: None
        // ┌▼──────────────────┐
        // │ First Node        │ <- update_from_client
        // └┬──────────────────┘
        //  │ Shard: Some(N)
        //  │ Ordering: Strong
        //  │ ShardKey: None
        //  │ ClockTag: None
        // ┌▼──────────────────┐
        // │ Leader node       │ <- update_from_peer
        // └┬──────────────────┘
        //  │ Shard: Some(N)
        //  │ Ordering: None(Weak)
        //  │ ShardKey: None
        //  │ ClockTag: { peer_id: IdOf(Leader node), clock_id: 1, clock_tick: 123 }
        // ┌▼──────────────────┐
        // │ Updating node     │ <- update_from_peer
        // └───────────────────┘

        let _update_rate_limiter = match &self.update_rate_limiter {
            Some(update_rate_limiter) => {
                // We only want to rate limit the first node in the chain
                if !shard_selector.is_shard_id() {
                    Some(update_rate_limiter.acquire().await)
                } else {
                    None
                }
            }

            None => None,
        };

        // TODO: `debug_assert(operation.clock_tag.is_none())` for `_update_shard_keys`/`update_from_client`!?

        let res = match shard_selector {
            ShardSelectorInternal::Empty => {
                collection
                    .update_from_client(
                        operation.operation,
                        wait,
                        timeout,
                        ordering,
                        None,
                        hw_measurement_acc.clone(),
                    )
                    .await?
            }

            ShardSelectorInternal::All => {
                let (sharding_method, shard_keys) = collection.get_sharding_method_and_keys().await;

                if shard_keys.is_empty() {
                    match sharding_method {
                        ShardingMethod::Custom => {
                            // No shards exist to apply the operation, but we acknowledge it
                            return Ok(UpdateResult {
                                operation_id: None,
                                status: UpdateStatus::Acknowledged,
                                clock_tag: operation.clock_tag,
                            });
                        }
                        ShardingMethod::Auto => {
                            collection
                                .update_from_client(
                                    operation.operation,
                                    wait,
                                    timeout,
                                    ordering,
                                    None,
                                    hw_measurement_acc.clone(),
                                )
                                .await?
                        }
                    }
                } else {
                    Self::_update_shard_keys(
                        &collection,
                        shard_keys,
                        operation.operation,
                        wait,
                        timeout,
                        ordering,
                        hw_measurement_acc.clone(),
                    )
                    .await?
                }
            }

            ShardSelectorInternal::ShardKey(shard_key) => {
                collection
                    .update_from_client(
                        operation.operation,
                        wait,
                        timeout,
                        ordering,
                        Some(shard_key),
                        hw_measurement_acc.clone(),
                    )
                    .await?
            }

            ShardSelectorInternal::ShardKeys(shard_keys) => {
                Self::_update_shard_keys(
                    &collection,
                    shard_keys,
                    operation.operation,
                    wait,
                    timeout,
                    ordering,
                    hw_measurement_acc.clone(),
                )
                .await?
            }

            ShardSelectorInternal::ShardKeyWithFallback(key) => {
                let shard_keys: Vec<_> = collection
                    .shards_holder()
                    .read()
                    .await
                    .route_with_fallback_for_write(key)?
                    .into_iter()
                    .map(|(_shard_ids, shard_key)| shard_key)
                    .collect();

                Self::_update_shard_keys(
                    &collection,
                    shard_keys,
                    operation.operation,
                    wait,
                    timeout,
                    ordering,
                    hw_measurement_acc.clone(),
                )
                .await?
            }
            ShardSelectorInternal::ShardId(shard_selection) => {
                collection
                    .update_from_peer(
                        operation,
                        shard_selection,
                        wait,
                        timeout,
                        ordering,
                        hw_measurement_acc.clone(),
                    )
                    .await?
            }
        };

        Ok(res)
    }

    async fn validate_lookup_from_collection_exists(
        &self,
        collection_name: &str,
    ) -> StorageResult<()> {
        match self.get_collection_unchecked(collection_name).await {
            Ok(_) => Ok(()),
            Err(StorageError::NotFound { .. }) => Err(StorageError::not_found(format!(
                "Collection {collection_name} not found"
            ))),
            Err(err) => Err(err),
        }
    }

    async fn validate_recommend_lookup_from(
        &self,
        request: &RecommendRequestInternal,
    ) -> StorageResult<()> {
        if let Some(lookup_from) = &request.lookup_from {
            self.validate_lookup_from_collection_exists(&lookup_from.collection)
                .await?;
        }
        Ok(())
    }

    async fn validate_query_lookup_from(
        &self,
        request: &CollectionQueryRequest,
    ) -> StorageResult<()> {
        if let Some(lookup_from) = &request.lookup_from {
            self.validate_lookup_from_collection_exists(&lookup_from.collection)
                .await?;
        }

        let mut prefetches: Vec<&CollectionPrefetch> = request.prefetch.iter().collect();
        while let Some(prefetch) = prefetches.pop() {
            if let Some(lookup_from) = &prefetch.lookup_from {
                self.validate_lookup_from_collection_exists(&lookup_from.collection)
                    .await?;
            }
            prefetches.extend(prefetch.prefetch.iter());
        }

        Ok(())
    }

    async fn validate_group_lookup_from(&self, request: &GroupRequest) -> StorageResult<()> {
        match &request.source {
            SourceRequest::Search(_) => {}
            SourceRequest::Recommend(request) => {
                self.validate_recommend_lookup_from(request).await?;
            }
            SourceRequest::Query(request) => {
                self.validate_query_lookup_from(request).await?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use collection::operations::CollectionUpdateOperations;
    use segment::types::{Filter, Payload, PointIdType, ShardKey};
    use shard::operations::payload_ops::{PayloadOps, SetPayloadOp};
    use shard::operations::point_ops::{PointInsertOperationsInternal, PointOperations};

    use super::TableOfContent;

    fn shard_key(name: &str) -> ShardKey {
        ShardKey::Keyword(name.into())
    }

    fn set_payload_op(point_ids: Vec<PointIdType>) -> CollectionUpdateOperations {
        CollectionUpdateOperations::PayloadOperation(PayloadOps::SetPayload(SetPayloadOp {
            payload: Payload::default(),
            points: Some(point_ids),
            filter: None,
            key: None,
        }))
    }

    /// Regression test for <https://github.com/qdrant/qdrant/issues/10064>.
    ///
    /// A `set_payload` request with `points: [9, 101]` and
    /// `shard_key: ["1", "2"]` must be split so shard key `"1"` only receives
    /// point `9` and shard key `"2"` only receives point `101`. Dispatching
    /// the full ID list to every shard key made each replica set report a
    /// false `No point with id ... found` for the ID owned by the other shard
    /// key, even though the update had already been applied there.
    #[test]
    fn test_split_set_payload_across_shard_keys() {
        let operation = set_payload_op(vec![PointIdType::NumId(9), PointIdType::NumId(101)]);
        let ids_per_key = vec![
            (shard_key("1"), HashSet::from([PointIdType::NumId(9)])),
            (shard_key("2"), HashSet::from([PointIdType::NumId(101)])),
        ];

        let split =
            TableOfContent::split_operation_per_shard_key(&operation, &ids_per_key).unwrap();

        assert_eq!(split.len(), 2);
        for (key, op) in split {
            let expected = match &key {
                ShardKey::Keyword(k) if k.as_str() == "1" => vec![PointIdType::NumId(9)],
                ShardKey::Keyword(k) if k.as_str() == "2" => {
                    vec![PointIdType::NumId(101)]
                }
                _ => panic!("unexpected shard key {key:?}"),
            };
            assert_eq!(
                op.point_ids(),
                Some(expected),
                "shard key {key:?} received foreign point ids"
            );
        }
    }

    /// A point ID that lives in none of the selected shard keys must be
    /// reported honestly as not found, not hidden behind partial writes.
    #[test]
    fn test_missing_point_id_reported_honestly() {
        let operation = set_payload_op(vec![PointIdType::NumId(9), PointIdType::NumId(404)]);
        let ids_per_key = vec![(shard_key("1"), HashSet::from([PointIdType::NumId(9)]))];

        let missed =
            TableOfContent::split_operation_per_shard_key(&operation, &ids_per_key).unwrap_err();
        assert_eq!(missed, PointIdType::NumId(404));
    }

    /// Deletes of missing points are a no-op: no honest not-found error and an
    /// empty split, so the caller can fall back to the previous dispatch.
    #[test]
    fn test_delete_points_keeps_noop_for_missing_ids() {
        let operation = CollectionUpdateOperations::PointOperation(PointOperations::DeletePoints {
            ids: vec![PointIdType::NumId(404)],
        });
        let ids_per_key = vec![(shard_key("1"), HashSet::new())];

        let split =
            TableOfContent::split_operation_per_shard_key(&operation, &ids_per_key).unwrap();
        assert!(split.is_empty());
    }

    /// Filter-based operations have no point IDs to split on: every shard key
    /// keeps receiving the full operation.
    #[test]
    fn test_filter_operation_is_not_split() {
        let operation = CollectionUpdateOperations::PayloadOperation(
            PayloadOps::ClearPayloadByFilter(Filter::default()),
        );
        let ids_per_key = vec![
            (shard_key("1"), HashSet::from([PointIdType::NumId(9)])),
            (shard_key("2"), HashSet::new()),
        ];

        let split =
            TableOfContent::split_operation_per_shard_key(&operation, &ids_per_key).unwrap();
        assert_eq!(split.len(), 2);
        for (_, op) in split {
            assert!(matches!(
                op,
                CollectionUpdateOperations::PayloadOperation(PayloadOps::ClearPayloadByFilter(_))
            ));
        }
    }

    /// Operations that can create points (upserts) must reach every shard key
    /// unsplit: a point may legitimately not exist anywhere yet.
    #[test]
    fn test_upsert_operation_is_not_split() {
        let operation = CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
            PointInsertOperationsInternal::PointsList(Vec::new()),
        ));
        let ids_per_key = vec![
            (shard_key("1"), HashSet::from([PointIdType::NumId(9)])),
            (shard_key("2"), HashSet::new()),
        ];

        let split =
            TableOfContent::split_operation_per_shard_key(&operation, &ids_per_key).unwrap();
        assert_eq!(split.len(), 2);
    }
}
