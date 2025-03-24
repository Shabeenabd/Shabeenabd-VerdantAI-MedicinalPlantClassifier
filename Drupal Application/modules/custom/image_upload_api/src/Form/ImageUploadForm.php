<?php

namespace Drupal\image_upload_api\Form;

use Drupal\Core\Form\FormBase;
use Drupal\Core\Form\FormStateInterface;
use GuzzleHttp\Client;
use Drupal\node\Entity\Node;
use Drupal\file\Entity\File;

class ImageUploadForm extends FormBase {

  public function getFormId() {
    return 'image_upload_form';
  }

  public function buildForm(array $form, FormStateInterface $form_state) {

    $form['#attached']['library'][] = 'image_upload_api/image_upload_api_styles';
    $form['#attached']['library'][] = 'image_upload_api/scroll_to_result';
    $form['image'] = [
      '#type' => 'managed_file',
      '#title' => $this->t('Upload or Take a Picture'),
      '#upload_validators' => [
        'file_validate_extensions' => ['jpg jpeg png'],
      ],
      '#upload_location' => 'public://uploads/',
      '#required' => TRUE,
    ];

    $form['submit'] = [
      '#type' => 'submit',
      '#value' => $this->t('Submit'),
      '#ajax' => [
        'callback' => '::ajaxSubmitCallback',
        'wrapper' => 'image-upload-result',
        'event' => 'click', // Ensure it's the click event
      ],
    ];

    $form['result'] = [
      '#type' => 'markup',
      '#markup' => '<div id="image-upload-result"></div>',
    ];

    return $form;
  }

  public function validateForm(array &$form, FormStateInterface $form_state) {
    parent::validateForm($form, $form_state);
  }

  public function submitForm(array &$form, FormStateInterface $form_state) {
    // Handled in the ajaxSubmitCallback.
  }

  public function ajaxSubmitCallback(array &$form, FormStateInterface $form_state) {
    $file = $form_state->getValue('image')[0];
    $file_entity = \Drupal::entityTypeManager()->getStorage('file')->load($file);
    $file_uri = $file_entity->getFileUri();
    $file_full_path = \Drupal::service('file_system')->realpath($file_uri);
    $client = new Client();
    $response = $client->request('POST', 'https://plantfastapi.icfoss.org/uploadfile/', [
        'multipart' => [
            [
                'name' => 'file',
                'contents' => fopen($file_full_path, 'r'),
            ],
        ],
    ]);

    $data = json_decode($response->getBody(), true);
   
    $prediction_id = $data['prediction'][0];
    $filename = $data['filename'];

    if ($prediction_id == -1) {
      $result = '<p>No details available for the given image.</p>';
      $form['result']['#markup'] = '<div id="image-upload-result">' . $result . '</div>';
      return $form['result'];
    }

    // Fetching node based on field_plant_id
    $query = \Drupal::entityQuery('node')
        ->condition('type', 'article')
        ->condition('field_plant_id', $prediction_id)
        ->condition('langcode', \Drupal::languageManager()->getCurrentLanguage()->getId())
        ->range(0, 1)
        ->accessCheck(FALSE); // Disable access check

    $nids = $query->execute();

    $result = '<h2>Prediction: ' . $data['prediction'][1] . '</h2>';

    if (!empty($nids)) {
        $node = \Drupal\node\Entity\Node::load(reset($nids));

        if ($node) {
            $title = $node->getTitle();
            $image =  $node->get('field_image');

            // Load the image file
            $image_file = File::load($node->get('field_image')->target_id);
            

            if ($image_file) {
              // Get the image URL with a desired image style (optional)
              $image_uri = $image_file->getFileUri(); 
            }
            
            $common_name = $node->get('field_plant_common_name')->value;
            $scientific_name = $node->get('field_plant_scientific_name')->value;
            $parts_used = $node->get('field_plant_parts_used')->value;
            $uses = $node->get('field_plant_uses')->value;
            $node_url = $node->toUrl()->toString();

            $result .= '<h3>' . $title . '</h3>';
            if ($image_uri) {
              // Convert the URI to a URL
              $image_url = \Drupal::service('file_url_generator')->generateAbsoluteString($image_uri);
              $result .= '<img src="' . $image_url . '" alt="' . $title . '">';
            }
            $result .= '<p><strong>Common Name:</strong> ' . $common_name . '</p>';
            $result .= '<p><strong>Scientific Name:</strong> ' . $scientific_name . '</p>';
            $result .= '<p><strong>Parts Used:</strong> ' . $parts_used . '</p>';
            $result .= '<p><strong>Uses:</strong> ' . $uses . '</p>';
            $result .= '<a href="' . $node_url . '" class="view-more-button">View More</a>';
        } else {
            $result .= '<p>No details available for the given prediction.</p>';
        }
    } else {
        $result .= '<p>No matching content found.</p>';
    }

    $form['result']['#markup'] = '<div id="image-upload-result">' . $result . '</div>';

    return $form['result'];
  }


}
