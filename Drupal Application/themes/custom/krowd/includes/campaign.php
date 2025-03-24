<?php
use CommerceGuys\Intl\Currency\CurrencyRepository;
use CommerceGuys\Intl\NumberFormat\NumberFormatRepository;
use CommerceGuys\Intl\Formatter\NumberFormatter;
use CommerceGuys\Intl\Formatter\CurrencyFormatter;
use Drupal\Core\Database\Query\Condition;

function krowd_preprocess_campaign(&$vars){
  $vars['#cache']['max-age'] = 0;
  $product_entity = $vars['product_entity'];
  $product = $vars['product'];
  
  $id = $product_entity->id();
  $raised = 0; // Current price
  $funded = 0; // % funded
  $start = 0;
  $goal = 0;
  $prefix = '';
  $suffix = '';
  $language = \Drupal::languageManager()->getCurrentLanguage()->getId();
  $video_link = '';

  $currency_code = 'USD';
  if($product_entity->hasField('field_campaign_start')){
    $start = $product_entity->get('field_campaign_start')->getValue();
    if(isset($start[0]['currency_code'])){
      $currency_code = $start[0]['currency_code'];
    }
    if(isset($start[0]['number'])){
      $start = $start[0]['number'];
    }
  }

  if($product_entity->hasField('field_campaign_video')){
    $video_link = $product_entity->get('field_campaign_video')->getValue();
    if(isset($video_link[0]['value'])){
      $video_link = $video_link[0]['value'];
    }
  }

  $currencyRepository = new CurrencyRepository;
  $numberFormatRepository = new NumberFormatRepository;

  $currencyFormatter = new CurrencyFormatter($numberFormatRepository, $currencyRepository, ['locale' => $language]);

  if($product_entity->hasField('field_campaign_goal')){
    $goal = $product_entity->get('field_campaign_goal')->getValue();
    if(isset($goal[0]['currency_code'])){
      $currency_code = $goal[0]['currency_code'];
    }
    if(isset($goal[0]['number'])){
      $goal = $goal[0]['number'];
    }
  }

  $results = Drupal::database()->select('{commerce_order_item}', 'oi');
  $results->leftJoin('{commerce_order}', 'o', 'oi.order_id = o.order_id');
  $results->leftJoin('{commerce_product_variation_field_data}', 'v', 'oi.purchased_entity = v.variation_id');
  $results->fields('v', array('product_id'));
  $results->fields('o', array('state'));
  $results->fields('oi', array('unit_price__number', 'total_price__number'));

  $results->condition(
    (new Condition('OR'))
      ->condition('o.state', 'payment_received', '=')
      ->condition('o.state', 'completed', '=')
  );
  $results->condition('v.product_id', $id, '=');
  $orders = $results->execute()->fetchAll(PDO::FETCH_ASSOC);

  foreach ($orders as $order) {
    $raised = $raised + $order['total_price__number'];
  }  
  $days_left = 0;
  if($product_entity->hasField('field_campaign_time')){
    $now = date('Y-m-d');
    $time_end = $product_entity->field_campaign_time->value;
    $days_left = round((strtotime($time_end) - strtotime($now)) / (60 * 60 * 24), 0);
  }
  if($days_left <= 0) {
    $days_left = t('expired');
     $vars['product']['variation_price'] = '';
    $vars['product']['variations'] = '<div class="expired">' . t('Expired Days left') . '</div>';
  }

  if($start) $raised = $raised + $start;

  if($goal > 0){
    $funded = round(($raised/$goal) * 100, 1);
  }
   $vars['funded_label'] = $funded;
  if($funded > 100) $funded = 100;
  $raised = preg_replace('/[^0-9,.]+/i', '', $raised);
  $raised = $currencyFormatter->format($raised, $currency_code, [ 'minimum_fraction_digits' => 0]);
  if($product_entity->hasField('title')){
    $title = $product_entity->get('title')->getValue();
    if(isset($title[0]['value'])){
      $title = $title[0]['value'];
    }
  }

  $vars['funded'] = $funded;
  $vars['raised'] = $raised;
  $vars['days_left'] = $days_left;
  $vars['video_link'] = $video_link;
  $vars['title'] = $title;
  return $vars;
}

function krowd_preprocess_commerce_product__campaign(&$vars){
  $vars = krowd_preprocess_campaign($vars);
}

function krowd_preprocess_commerce_product__campaign__teaser(&$vars){
  $vars = krowd_preprocess_campaign($vars);
}

function krowd_preprocess_commerce_product__campaign__teaser_2(&$vars){
  $vars = krowd_preprocess_campaign($vars);
}

function krowd_preprocess_commerce_product__campaign__teaser_3(&$vars){
  $vars = krowd_preprocess_campaign($vars);
}