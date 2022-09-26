 # Pairwise loss
  dot_product = tf.matmul(class_vectors_mat, tf.transpose(class_vectors_mat))
  square_norm = tf.diag_part(dot_product)
  distances = tf.expand_dims(square_norm, 0) - 2.0 * ( dot_product) + tf.expand_dims(square_norm, 1)
  distances = tf.maximum(distances, 0.0)
  mask = tf.to_float(tf.equal(self.distances, 0.0))
  distance = distances + mask * 1e-16
  distance = tf.sqrt(distance)
  distance = distance * (1.0 -mask)
  pairwise_dist = (-tf.reduce_sum(distance))*lambd_pair
  
  # Attractive loss
  normalize_class_vec = tf.nn.l2_normalize(class_vec,1)        
  normalize_sent_vec = tf.nn.l2_normalize(sent_vec ,1)      
  dist_cos = 1-tf.reduce_sum(tf.multiply(normalize_class_vec,normalize_sent_vec),1)
  attractive_loss = tf.reduce_mean(dist_cos)*lambd_attr

  
  # Repulsive term
  class_vectors_for_rep = tf.nn.embedding_lookup(class_vectors,preds_not_related)
  normalize_rep = tf.nn.l2_normalize(class_vectors_for_rep,1)        
  cos_rep = 1-tf.reduce_sum(tf.multiply(normalize_sent_vec,normalize_rep),1)
  repulsive_term =-tf.reduce_mean(cos_rep )*lambda_term
