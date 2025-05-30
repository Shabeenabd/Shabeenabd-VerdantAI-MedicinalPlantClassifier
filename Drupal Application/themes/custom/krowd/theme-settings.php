<?php
use Drupal\Core\Extension\Extension;
use Drupal\Core\Extension\ExtensionDiscovery;

use Drupal\system\Controller\ThemeController;
use Drupal\Core\Theme\ThemeManagerInterface;
use Symfony\Component\HttpFoundation\Request;

/**
 * Implementation of hook_form_system_theme_settings_alter()
 *
 * @param $form
 *   Nested array of form elements that comprise the form.
 *
 * @param $form_state
 *   A keyed array containing the current state of the form.
 */
function krowd_form_system_theme_settings_alter(&$form, &$form_state) {
  $form['#attached']['library'][] = 'krowd/gavias-charityplus-admin';  
  // Get the build info for the form
  $build_info = $form_state->getBuildInfo();
  // Get the theme name we are editing
  $theme = \Drupal::theme()->getActiveTheme()->getName();
  // Create Omega Settings Object

  $form['core'] = array(
    '#type' => 'vertical_tabs',
    '#attributes' => array('class' => array('entity-meta')),
    '#weight' => -899
  );

  $form['theme_settings']['#group'] = 'core';
  $form['logo']['#group'] = 'core';
  $form['favicon']['#group'] = 'core';

  $form['theme_settings']['#open'] = FALSE;
  $form['logo']['#open'] = FALSE;
  $form['favicon']['#open'] = FALSE;
  
  // Custom settings in Vertical Tabs container
  $form['options'] = array(
    '#type' => 'vertical_tabs',
    '#attributes' => array('class' => array('entity-meta')),
    '#weight' => -999,
    '#default_tab' => 'edit-variables',
    '#states' => array(
      'invisible' => array(
       ':input[name="force_subtheme_creation"]' => array('checked' => TRUE),
      ),
    ),
  );

  /* --------- Setting general ----------------*/
  $form['general'] = array(
    '#type' => 'details',
    '#attributes' => array(),
    '#title' => t('Gerenal options'),
    '#weight' => -999,
    '#group' => 'options',
    '#open' => FALSE,
  );

  $form['general']['sticky_menu'] =array(
    '#type' => 'select',
    '#title' => t('Enable Sticky Menu'),
    '#default_value' => theme_get_setting('sticky_menu'),
    '#group' => 'general',
    '#options' => array(
      '0'        => t('Disable'),
      '1'        => t('Enable')
     ) 
  ); 
  
  $form['general']['site_layout'] = array(
    '#type' => 'select',
    '#title' => t('Body Layout'),
    '#default_value' => theme_get_setting('site_layout'),
    '#options' => array(
      'wide' => t('Wide (default)'),
      'boxed' => t('Boxed'),
    ),
  );

  $form['general']['preloader'] = array(
    '#type' => 'select',
    '#title' => t('Preloader'),
    '#default_value' => theme_get_setting('preloader'),
    '#group' => 'options',
    '#options' => array(
      '0' => t('Disable'),
      '1' => t('Enable')
    ),
  );
  $form['general']['google_api_key'] = array(
    '#type' => 'textfield',
    '#title' => t('Google Map Api'),
    '#default_value' => theme_get_setting('google_api_key')
  );

  
  /*--------- Setting Header ------------ */
  $form['header'] = array(
    '#type' => 'details',
    '#attributes' => array(),
    '#title' => t('Header options'),
    '#weight' => -998,
    '#group' => 'options',
    '#open' => FALSE,
  );

  $form['header']['default_header'] = array(
    '#type' => 'select',
    '#title' => t('Setting default header'),
    '#default_value' => theme_get_setting('default_header'),
    '#options' => array(
      'header' => t('Header default'),
      'header-1' => t('Header v1'),
      'header-2' => t('header v2')
    ),
  );

  $form['header']['link_btn_header'] = array(
    '#title'  => t('Link Button Header'),
    '#type' => 'textfield',
    '#default_value' => theme_get_setting('link_btn_header'),
  );

    $form['header']['text_btn_header'] = array(
    '#title'  => t('Text Button Header'),
    '#type' => 'textfield',
    '#default_value' => theme_get_setting('text_btn_header') ? theme_get_setting('text_btn_header') : t('Donate Now'),
  );

  // User CSS
  $form['options']['css_customize'] = array(
    '#type' => 'details',
    '#attributes' => array(),
    '#title' => t('Customize css'),
    '#weight' => -996,
    '#group' => 'options',
    '#open' => TRUE,
);

  /*--------- Setting Footer ------------ */
  $form['footer'] = array(
    '#type' => 'details',
    '#attributes' => array(),
    '#title' => t('Footer options'),
    '#weight' => -998,
    '#group' => 'options',
    '#open' => FALSE,
  );

  $form['footer']['footer_skin'] = array(
    '#type' => 'select',
    '#title' => t('Footer Skin'),
    '#default_value' => theme_get_setting('footer_skin'),
    '#group' => 'footer',
    '#options' => array(
      ''       => t('Footer Default'),
    ),
  );
  $form['footer']['footer_first_size'] = array(
    '#type' => 'select',
    '#title' => t('Footer First Size'),
    '#default_value' => theme_get_setting('footer_first_size') ? theme_get_setting('footer_first_size') : 3,
    '#options' => array(0 => t('Hidden'), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12),
    '#description' => 'Setting width for grid boostrap / 12'
  );

  $form['footer']['footer_second_size'] = array(
    '#type' => 'select',
    '#title' => t('Footer Second Size'),
    '#default_value' => theme_get_setting('footer_second_size') ? theme_get_setting('footer_second_size') : 3,
    '#options' => array(0 => t('Hidden'), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12),
    '#description' => 'Setting width for grid boostrap / 12'
  );

  $form['footer']['footer_third_size'] = array(
    '#type' => 'select',
    '#title' => t('Footer Third Size'),
    '#default_value' => theme_get_setting('footer_third_size') ? theme_get_setting('footer_third_size') : 3,
    '#options' => array(0 => t('Hidden'), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12),
    '#description' => 'Setting width for grid boostrap / 12'
  );

  $form['footer']['footer_four_size'] = array(
    '#type' => 'select',
    '#title' => t('Footer Four Size'),
    '#default_value' => theme_get_setting('footer_four_size') ? theme_get_setting('footer_four_size') : 3,
    '#options' => array(0 => t('Hidden'), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12),
    '#description' => 'Setting width for grid boostrap / 12'
  );
  // User CSS
  $form['options']['css_customize'] = array(
    '#type' => 'details',
    '#attributes' => array(),
    '#title' => t('Customize css'),
    '#weight' => -996,
    '#group' => 'options',
    '#open' => TRUE,
  );

  $form['customize']['customize_css'] = array(
    '#type' => 'textarea',
    '#title' => t('Add your own CSS'),
    '#group' => 'css_customize',
    '#attributes' => array('class' => array('code_css') ),
    '#default_value' => theme_get_setting('customize_css'),
  );
  
  //Customize color ----------------------------------
  $form['options']['settings_customize'] = array(
    '#type' => 'details',
    '#attributes' => array(),
    '#title' => t('Settings Customize'),
    '#weight' => -997,
    '#group' => 'options',
    '#open' => TRUE,
  );
  
  $form['options']['settings_customize']['settings'] = array(
    '#type' => 'details',
    '#open' => TRUE,
    '#attributes' => array(),
    '#title' => t('Cutomize Setting'),
    '#weight' => -999,
  );

  $form['options']['settings_customize']['settings']['enable_customize'] = array(
    '#type' => 'select',
    '#title' => t('Enable Display Cpanel Customize'),
    '#default_value' => theme_get_setting('enable_customize'),
    '#group' => 'settings',
    '#options' => array(
      '0'        => t('Disable'),
      '1'        => t('Enable'),
    ),
  );

  $form['actions']['submit']['#value'] = t('Save');
} 
